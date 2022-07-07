import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.datasets.transforms as transforms
from lib.datasets import person_search_dataset, query_dataset
from lib.models.baseline import PersonSearch
from lib.utils import Logger
from lib.utils import collect_fn
from lib.utils import evaluate


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--test_box_score_thresh', default=0.5, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--use_saved_result', default=0, type=int)
    parser.add_argument('--dataset_name', default='sysu', type=str)
    args = parser.parse_args()
    return args


def extract_feats(model, query_loader, gallery_dataloader, output_dir):
    gallery_feats_dict = {}
    query_feats = []
    with torch.no_grad():
        for i, data in enumerate(
                tqdm(gallery_dataloader, total=len(gallery_dataloader), desc='extracting gallery features')):
            imgs, targets = data
            imgs = [img.cuda() for img in imgs]
            feats_dict = model(imgs, targets)
            gallery_feats_dict.update(feats_dict)

        for i, data in tqdm(enumerate(query_loader), desc='extracting query features', total=len(query_loader)):
            imgs, targets = data
            imgs = [img.cuda() for img in imgs]
            feats = model(imgs, targets, query=True)
            query_feats.append(feats.cpu())
        query_feats = torch.cat(query_feats, dim=0)
    test_feats = {'query': query_feats, 'gallery': gallery_feats_dict}
    save_dir = os.path.join(output_dir, 'test_feats.pkl')
    torch.save(test_feats, save_dir)
    return test_feats


def main():
    """Test the model"""
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    logs_dir = os.path.dirname(args.resume)
    sys.stdout = Logger(os.path.join(logs_dir, 'test.txt'))
    # Compose transformations for the dataset
    transformer = transforms.Compose([transforms.ToTensor()])
    # Load the dataset
    query_set = query_dataset(args.data_dir, args.dataset_name, transform=transformer)
    test_set = person_search_dataset(args.data_dir, args.dataset_name, 'test', transform=transformer)
    if args.use_saved_result:
        test_feats = torch.load(os.path.join(logs_dir, 'test_feats.pkl'))
    else:
        reid_classes = 483 if args.dataset_name == 'prw' else 5532
        model_dict = torch.load(args.resume, map_location='cpu')['state_dict']
        if list(model_dict.keys())[0].startswith('module.'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        model = PersonSearch(num_classes=reid_classes, min_size=640, max_size=960,
                             test_box_thresh=args.test_box_score_thresh)
        model.load_state_dict(model_dict)
        model.eval()
        model.cuda()
        query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, collate_fn=collect_fn, pin_memory=True)
        dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, collate_fn=collect_fn, pin_memory=True)
        test_feats = extract_feats(model, query_loader, dataloader, logs_dir)

    gallery_size = None if args.dataset_name == 'prw' else [100]
    num_gt = query_set.query_file.loc[:, 'num_gt'].tolist()
    query_feats, gallery_feats = test_feats['query'], test_feats['gallery']
    query_info = query_set.query_id, query_set.query_imname, query_set.query_cam
    test_set.evaluate_detections(gallery_feats, det_thresh=args.test_box_score_thresh)
    evaluate(args.dataset_name, query_feats, query_info, gallery_feats, num_gt, gallery_size=gallery_size)


if __name__ == '__main__':
    main()
