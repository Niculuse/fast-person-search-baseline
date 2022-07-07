import os

import PIL
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from torchvision.ops.boxes import box_iou
from tqdm import tqdm

from lib.datasets import process_prw, process_sysu


class person_search_dataset(Dataset):
    def __init__(self, root_dir, dataset_name, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, 'frames')
        self.anno_dir = os.path.join(self.root_dir, 'processed_annotations')
        self.imnames_file = '{}ImnamesSe.csv'.format(self.split)

        if not os.path.exists(self.anno_dir):
            if dataset_name == 'prw':
                process_prw.preprocess(root_dir)
            elif dataset_name == 'sysu':
                process_sysu.preprocess(root_dir)

        self.imnames = pd.read_csv(os.path.join(
            self.anno_dir, self.imnames_file), header=None, squeeze=True)
        new_format_data = os.path.join(self.root_dir, '{}_data.torch'.format(self.split))
        if os.path.exists(new_format_data):
            self.data = torch.load(new_format_data)
        else:
            self.all_boxes_file = '{}AllDF.csv'.format(self.split)
            self.query_file = 'queryDF.csv'
            self.all_boxes = pd.read_csv(os.path.join(
                self.anno_dir, self.all_boxes_file))
            self.all_boxes['del_x'] += self.all_boxes['x1']
            self.all_boxes['del_y'] += self.all_boxes['y1']
            self.all_boxes.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)
            self.data = self.make_data(new_format_data)
        self.num_classes = 483 if dataset_name == 'prw' else 5532

    def __len__(self):
        return self.imnames.shape[0]

    def __getitem__(self, index):

        img_name = self.imnames[index]
        im_path = os.path.join(self.image_dir, img_name)
        pids = self.data[img_name]['pids']
        boxes = self.data[img_name]['boxes']
        img = PIL.Image.open(im_path)

        if self.transform is not None:
            img, boxes = self.transform(img, boxes)

        labels = torch.ones_like(pids)
        target = {'boxes': boxes, 'labels': labels, 'pids': pids, 'img_name': img_name}
        return img, target

    def evaluate_detections(self, gallery_det, det_thresh=0.5, iou_thresh=0.5):
        """evaluate the performance of the person detector"""
        assert self.imnames.shape[0] == len(gallery_det)
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for k in range(len(gallery_det)):
            img_name = self.imnames.iloc[k]
            gt_boxes = self.data[img_name]['boxes']
            det = gallery_det[img_name]['boxes']
            if det.shape[0] == 0:
                continue
            inds = det[:, 4] >= det_thresh
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = box_iou(gt_boxes, det[:, :4]).numpy()
            tfmat = ious >= iou_thresh
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        print('The performance of person detector is:')
        print('  Recall = {:.2%}'.format(det_rate))
        print('  AP = {:.2%}'.format(ap))
        return det_rate, ap

    def make_data(self, save_dir):
        data = {}
        for i in tqdm(range(len(self.imnames)), desc='making new data format'):
            img_name = self.imnames[i]
            boxes_df = self.all_boxes.query('imname==@img_name')
            boxes = boxes_df.loc[:, 'x1': 'pid'].copy()
            boxes = boxes.values.astype(np.float32)
            pids = torch.from_numpy(boxes[:, -1]).long()
            boxes = torch.from_numpy(boxes[:, :4]).float()
            data[img_name] = {'pids': pids, 'boxes': boxes}
        torch.save(data, save_dir)
        return data


class query_dataset(Dataset):
    def __init__(self, root_dir, dataset_name, transform=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, 'frames')
        query_file_dir = os.path.join(self.root_dir, 'processed_annotations', 'queryDF.csv')
        self.query_file = pd.read_csv(query_file_dir)
        self.query_id = self.query_file.loc[:, 'pid'].tolist()
        self.query_imname = self.query_file.loc[:, 'imname'].tolist()
        self.query_cam = [int(name[1]) for name in self.query_imname]
        boxes = self.query_file.loc[:, 'x1':'del_y'].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        self.query_box = boxes

    def __len__(self):
        return len(self.query_file)

    def __getitem__(self, index):
        data = self.query_file.loc[index]
        img_name = data['imname']
        im_path = os.path.join(self.image_dir, img_name)
        boxes = data['x1':'del_y'].values.astype(np.float32)
        boxes[2:] += boxes[:2]
        boxes = torch.tensor(boxes).float().view(1, 4)
        img = PIL.Image.open(im_path)
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)
        target = {'boxes': boxes, 'img_name': img_name}
        return img, target
