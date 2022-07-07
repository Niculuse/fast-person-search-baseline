import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.cuda.amp import autocast
from torch.jit.annotations import Tuple, List, Dict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou, RoIAlign

from .model_utils import base_net, id_net


class PersonSearch(nn.Module):
    def __init__(self, backbone_name, num_classes, min_size=640, max_size=960, test_box_thresh=0.5):
        super(PersonSearch, self).__init__()
        end_layer = 2
        self.test_box_thresh = test_box_thresh
        self.end_layer = end_layer
        backbone = base_net(backbone_name=backbone_name, end_layer=self.end_layer, pretrained=True)
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=((4, 8, 16, 32),), aspect_ratios=((1.0, 2.0, 3.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, 2)
        box_head = TwoMLPHead(backbone.out_channels * roi_pooler.output_size[0] * roi_pooler.output_size[1],
                              representation_size)
        self.faster_rcnn = FasterRCNN(backbone, num_classes=None,
                                      # transform parameters
                                      min_size=min_size, max_size=max_size,
                                      image_mean=None, image_std=None,
                                      # RPN parameters
                                      rpn_anchor_generator=anchor_generator, rpn_head=None,
                                      rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                                      rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                                      rpn_nms_thresh=0.7,
                                      rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                      rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                      # Box parameters
                                      box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor,
                                      box_score_thresh=test_box_thresh, box_nms_thresh=0.3, box_detections_per_img=100,
                                      box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                                      box_batch_size_per_image=128, box_positive_fraction=0.25,
                                      bbox_reg_weights=None)
        stride = 2 ** (self.end_layer + 1)
        self.RoIAlign = RoIAlign(output_size=(256 // stride, 128 // stride), spatial_scale=1.0 / stride,
                                 sampling_ratio=2, aligned=True)
        self.id_net = id_net(backbone_name=backbone_name, num_classes=num_classes, last_stride=1,
                             start_layer=self.end_layer + 1, pretrained=True)
        self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

    @autocast()
    def forward(self, images, targets=None, query=False):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.faster_rcnn.transform(images, targets)
        features = self.faster_rcnn.backbone(images.tensors)
        if query:
            boxes = [t['boxes'].cuda() for t in targets]
            roi_feats = self.RoIAlign(features, boxes)

            roi_feats = self.id_net(roi_feats)
            normed_feats = F.normalize(roi_feats)
            return normed_feats.cpu()
        # detection part
        rpn_feats = features
        if isinstance(rpn_feats, torch.Tensor):
            rpn_feats = OrderedDict([('0', rpn_feats)])
        proposals, proposal_losses = self.faster_rcnn.rpn(images, rpn_feats, targets if self.training else None)
        detections, detector_losses = self.faster_rcnn.roi_heads(rpn_feats, proposals, images.image_sizes,
                                                                 targets if self.training else None)
        losses, data = {}, {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # identification part
        if self.training:
            # proposals, labels = self.select_training_samples(proposals, targets)
            proposals, labels = [], []
            for i in range(len(targets)):
                proposals.append(targets[i]['boxes'])
                labels.append(targets[i]['pids'].to(images.tensors.device))
            roi_feats = self.RoIAlign(features, proposals)

            roi_feats, roi_logits = self.id_net(roi_feats)
            labels = torch.cat(labels)

            cls_loss = self.id_loss(roi_logits, labels)

            pred = (roi_logits.max(1)[1] == labels)[labels > -1]
            pred = pred.cpu()
            losses.update({'loss_id': cls_loss})
            return losses, pred
        else:
            detect_boxes = [d['boxes'] for d in detections]
            detections = self.faster_rcnn.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            for i in range(len(detections)):
                img_name = targets[i]['img_name']
                detection = detect_boxes[i]
                scores = detections[i]['scores']
                if detection.shape[0] == 0:
                    data[img_name] = {'feat': torch.tensor([]), 'id': torch.tensor([]), 'boxes': torch.tensor([])}
                    continue

                gt_box = targets[i]['boxes'].cuda()
                gt_pid = targets[i]['pids'].cuda()
                IoUs = box_iou(detection, gt_box)
                values, indices = torch.max(IoUs, dim=1)
                label = -torch.ones(detection.shape[0], dtype=torch.long)
                valid = values > self.test_box_thresh
                label[valid] = gt_pid[indices][valid].cpu()

                roi_feats = self.RoIAlign(features[i][None], [detection])

                roi_feats = self.id_net(roi_feats)
                normed_feats = F.normalize(roi_feats)
                data[img_name] = {'feat': normed_feats.cpu(), 'id': label,
                                  'boxes': torch.cat((detections[i]['boxes'], scores.view(-1, 1)), dim=1).cpu()}
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, data)

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        assert targets is not None
        dtype = proposals[0].dtype

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["pids"] + 2 for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.faster_rcnn.roi_heads.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.faster_rcnn.roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.faster_rcnn.roi_heads.subsample(labels)
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds] - 2
            valid = labels[img_id] >= -1
            proposals[img_id] = proposals[img_id][valid]
            labels[img_id] = labels[img_id][valid].cuda()
        return proposals, labels
