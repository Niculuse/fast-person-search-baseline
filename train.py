import argparse
import os
import os.path as osp
import sys
import time
import warnings
from collections import defaultdict

import torch
from torch.cuda.amp import autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

import lib.datasets.transforms as transforms
from lib.datasets import person_search_dataset, query_dataset
from lib.models.baseline import PersonSearch
from lib.utils import Logger, loss_plot, evaluate, collect_fn
from test import extract_feats

warnings.filterwarnings('ignore')


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Training the Person Search Model')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--backbone', default='resnet50_ibn_a', type=str)
    parser.add_argument('--pretrained_path', default='', type=str)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--test_box_score_thresh', default=0.5, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--logs_dir', default='logs', type=str)
    parser.add_argument('--dataset', default='prw', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    return args


def train_model(dataloader, model, optimizer, num_epochs, save_dir, start_epoch, args, local_rank=0):
    """Train the model"""
    dataset_len = len(dataloader)
    model.train()
    hist = defaultdict(list)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [12], gamma=0.1, last_epoch=start_epoch - 1, verbose=True)
    start = time.time()
    best_mAP = best_top1 = 0.0

    for epoch in range(start_epoch, num_epochs):
        if torch.cuda.device_count() > 1:
            dataloader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        for step, data in enumerate(dataloader):
            iter_start = time.time()
            imgs, targets = data
            imgs = [img.cuda(non_blocking=True) for img in imgs]
            for target in targets:
                target['boxes'] = target['boxes'].cuda(non_blocking=True)
                target['labels'] = target['labels'].cuda(non_blocking=True)
            with autocast():
                loss_dict, pred = model(imgs, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_end = time.time()

            loss_rpn_cls = loss_dict['loss_objectness'].mean().item()
            loss_rpn_bbox = loss_dict['loss_rpn_box_reg'].mean().item()
            loss_cls = loss_dict['loss_classifier'].mean().item()
            loss_bbox = loss_dict['loss_box_reg'].mean().item()
            loss_id = loss_dict['loss_id'].mean().item()

            acc = pred.sum().float() / pred.shape[0]

            hist['loss_rpn_cls'].append(loss_rpn_cls)
            hist['loss_rpn_bbox'].append(loss_rpn_bbox)
            hist['loss_cls'].append(loss_cls)
            hist['loss_bbox'].append(loss_bbox)
            hist['loss_id'].append(loss_id)

            hist['acc'].append(acc)
            if step % args.print_freq == 0:
                length = len(hist['loss_cls'])
                print('Epoch: [{:2d}/{:2d}] iter: [{:4d}/{:4d}] '
                      'rpn_cls:{:.3f}({:.3f})  rpn_box:{:.3f}({:.3f}) '
                      'cls:{:.3f}({:.3f})  box:{:.3f}({:.3f}) '
                      'id: {:.3f}({:.3f})'
                      'acc: {:.2%} time: {:.2f}s'
                      .format(epoch + 1, num_epochs, step, dataset_len,
                              loss_rpn_cls, sum(hist['loss_rpn_cls']) / length,
                              loss_rpn_bbox, sum(hist['loss_rpn_bbox']) / length,
                              loss_cls, sum(hist['loss_cls']) / length,
                              loss_bbox, sum(hist['loss_bbox']) / length,
                              loss_id, sum(hist['loss_id']) / length,
                              acc, iter_end - iter_start))
        epoch_end = time.time()
        scheduler.step()
        print('\nEntire epoch time cost: {:.2f} hours\n'.format(
            (epoch_end - epoch_start) / 3600))
        # Save the trained model after each epoch
        save_name = os.path.join(save_dir, 'checkpoint.pth')
        if local_rank == 0:
            state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
            state_dict = {'epoch': epoch + 1, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
            torch.save(state_dict, save_name)
            if (epoch >= 12) & (((epoch + 1) % 2 == 0) | (epoch + 1 == num_epochs)):
                recall, ap, mAP, top1 = eval(model, args, save_dir)
                better_map = mAP >= best_mAP
                if better_map:
                    best_recall = recall
                    best_ap = ap
                    best_mAP = mAP
                    best_top1 = top1
                    best_epoch = epoch
                    state_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'recall': best_recall, 'ap': best_ap}
                    torch.save(state_dict, os.path.join(save_dir, 'model_best.pth'))

                state_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'recall': recall, 'ap': ap, 'mAP': mAP, 'top1': top1}
                torch.save(state_dict, os.path.join(save_dir, 'epoch_{:d}.pth'.format(epoch + 1)))
                print('the best is recall:{:.2%}, ap:{:.2%}, mAP:{:.1%}, top1:{:.1%} in epoch {:d}'.format(
                    best_recall, best_ap, best_mAP, best_top1, best_epoch + 1))
    end = time.time()
    if local_rank == 0:
        loss_plot(hist, save_dir)
    print('train process finished!!! time cost: {:.2f} hours'.format((end - start) / 3600))


def eval(model, args, save_dir):
    model.eval()
    test_transformer = transforms.Compose([transforms.ToTensor()])
    query_set = query_dataset(args.data_dir, args.dataset, transform=test_transformer)
    test_set = person_search_dataset(args.data_dir, args.dataset, 'test', transform=test_transformer)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=collect_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=collect_fn, pin_memory=True)
    test_feats = extract_feats(model, query_loader, test_loader, save_dir)
    gallery_size = None if args.dataset == 'prw' else [100]
    num_gt = query_set.query_file.loc[:, 'num_gt'].tolist()
    query_feats, gallery_feats = test_feats['query'], test_feats['gallery']
    recall, ap = test_set.evaluate_detections(gallery_feats, det_thresh=args.test_box_score_thresh)
    query_info = query_set.query_id, query_set.query_imname, query_set.query_cam
    mAP, top1 = evaluate(args.dataset, query_feats, query_info, gallery_feats, num_gt, gallery_size=gallery_size)
    return recall, ap, mAP, top1


def main():
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    Time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    save_dir = os.path.join(args.logs_dir, args.dataset, args.backbone, Time)
    args.data_dir = osp.join(args.data_dir, args.dataset)
    # Compose transformations for the dataset
    train_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    # Load the dataset
    train_set = person_search_dataset(args.data_dir, args.dataset, 'train', train_transformer)
    reid_classes = train_set.num_classes
    model = PersonSearch(args.backbone, reid_classes, min_size=640, max_size=960,
                         test_box_thresh=args.test_box_score_thresh)

    # Choose parameters to be updated during training
    lr = args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=0.9, lr=lr, weight_decay=1e-4, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    else:
        raise KeyError(args.optimizer)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint['epoch']
        print('Resuming model checkpoint: {}, epoch: {}'.format(args.resume, start_epoch))
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0

    # distributed parallel training
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=False)
        train_sampler = DistributedSampler(train_set)
        shuffle = False
    else:
        local_rank = 0
        train_sampler = None
        shuffle = True if train_sampler is None else False
        device = torch.device('cuda')
        model.to(device)
    if args.resume:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler,
                              drop_last=True, num_workers=8, collate_fn=collect_fn, pin_memory=True)
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sys.stdout = Logger(osp.join(save_dir, 'log.txt'))
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        print('Trained models will be save to', os.path.abspath(save_dir))
    # Train the model
    train_model(train_loader, model, optimizer, args.epochs, save_dir, start_epoch, args, local_rank)


if __name__ == '__main__':
    main()
