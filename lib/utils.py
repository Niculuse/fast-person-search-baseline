import errno
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def collect_fn(data):
    imgs, targets = [], []
    for d in data:
        imgs.append(d[0])
        targets.append(d[1])
    return imgs, targets


def collect_fn_for_resample(data):
    imgs, targets = [], []
    pos_imgs, pos_targets = [], []
    for d in data:
        imgs.append(d[0])
        targets.append(d[1])
        pos_imgs.extend(d[2])
        pos_targets.extend(d[3])
    imgs.extend(pos_imgs)
    targets.extend(pos_targets)
    return imgs, targets


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def loss_plot(items, path):
    mkdir_if_missing(path)
    items_keys = []
    for key in items:
        items_keys.append(key)
    x = range(len(items[items_keys[0]]))
    for key, values in items.items():
        y = items[key]
        plt.plot(x, y, label=key)

    plt.xlabel('epoch')
    plt.ylabel('Loss')

    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, 'losses.png')
    plt.savefig(path)
    plt.close()


def fix_bn(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def evaluate(data_name, query_feats, query_info, gallery, num_gt, top_k=10, cmc_topk=(1, 5, 10), gallery_size=None,
             cross_camera=False):
    if gallery_size is not None:
        q_g_file_dirs = [os.path.join('data', data_name, 'processed_annotations', 'q_to_g' + str(size) + 'DF.csv') for
                         size in
                         gallery_size]
        q_g_files = [pd.read_csv(q_g_dir) for q_g_dir in q_g_file_dirs]
    else:
        q_g_files = None
    query_ids, query_imname, query_cam = query_info
    query_ids = torch.tensor(query_ids)
    query_cam = torch.tensor(query_cam)
    query_feats = query_feats.cuda()

    if gallery_size == None:
        gallery_feats, gallery_pids, gallery_cams, gallery_imnames = [], [], [], []
        for k, v in gallery.items():
            cam_id = [int(k[1])] * v['id'].shape[0]
            gallery_feats.append(v['feat'].cuda())
            gallery_pids.append(v['id'])
            gallery_cams.extend(cam_id)
            gallery_imnames.extend([k] * v['id'].shape[0])
        gallery_feats = torch.cat(gallery_feats, dim=0)
        gallery_cams = torch.tensor(gallery_cams)
        gallery_pids = torch.cat(gallery_pids, dim=0)
        sim_mat = query_feats.mm(gallery_feats.t())
        orders = torch.argsort(sim_mat, dim=1, descending=True).cpu()
        sim_mat = sim_mat.cpu()
        matches = query_ids[:, None] == gallery_pids[orders]
        if cross_camera:
            valid_mat = query_cam[:, None] != gallery_cams[orders]
        else:
            query_imname = np.array(query_imname)
            gallery_imnames = np.array(gallery_imnames)
            valid_mat = query_imname[:, np.newaxis] != gallery_imnames[orders]
        matches_mat = torch.zeros_like(sim_mat)
        aps, recall_query = [], []

        for i in tqdm(range(query_feats.shape[0]), desc='calculating accuracy'):
            match = matches[i]
            score = sim_mat[i][orders[i]]
            valid = valid_mat[i]
            y_true = match[valid]
            y_score = score[valid]
            recall = y_true.sum(dtype=torch.float) / num_gt[i]
            if not torch.any(y_true):
                recall_query.append(0.0)
                aps.append(0.0)
                continue
            ap = (average_precision_score(y_true, y_score) * recall).item()
            aps.append(ap)
            recall_query.append(recall)
            matches_mat[i, :y_true.shape[0]] = y_true

        mAP = torch.tensor(aps).mean()
        match_mat = matches_mat[:, :top_k]
        match_mat = match_mat.cumsum(dim=1)
        match_mat = match_mat >= 1
        cmc_scores = match_mat.float().mean(dim=0)
        print('recall of query persons is {:.1%}'.format(sum(recall_query) / len(recall_query)))
        print('  mAP:{:4.1%}'.format(mAP))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))

    else:
        for i, size in enumerate(gallery_size):
            q_g_file = q_g_files[i]
            matches_mat = torch.zeros((q_g_file.shape[0], top_k))
            aps, recall_query = [], []

            for j in tqdm(range(q_g_file.shape[0]), desc='calculating accuracy'):
                q_feat = query_feats[j].unsqueeze(0)
                q_id = query_ids[j]
                test_images = q_g_file.loc[j].tolist()[1:]
                gallery_feats, gallery_ids = [], []
                for img_name in test_images:
                    gallery_feats.append(gallery[img_name]['feat'])
                    gallery_ids.append(gallery[img_name]['id'])
                gallery_feats = torch.cat(gallery_feats, dim=0).cuda()
                gallery_ids = torch.cat(gallery_ids, dim=0)
                sim_mat = q_feat.mm(gallery_feats.t())[0]
                order = torch.argsort(sim_mat, dim=0, descending=True).cpu()
                sim_mat = sim_mat.cpu()
                y_score = sim_mat[order]
                match = q_id == gallery_ids[order]
                y_true = match
                recall = y_true.sum(dtype=torch.float) / num_gt[j]
                if not torch.any(y_true):
                    recall_query.append(0.0)
                    aps.append(0.0)
                    continue

                ap = (average_precision_score(y_true, y_score) * recall).item()
                aps.append(ap)
                recall_query.append(recall)
                matches_mat[j] = y_true[:top_k]

            mAP = torch.tensor(aps).mean()
            match_mat = matches_mat[:, :top_k]
            match_mat = match_mat.cumsum(dim=1)
            match_mat = match_mat >= 1
            cmc_scores = match_mat.float().mean(dim=0)
            print('gallery size: {:d}'.format(size))
            print('recall of query persons is {:.1%}'.format(sum(recall_query) / len(recall_query)))
            print('  mAP:{:4.1%}'.format(mAP))
            for k in cmc_topk:
                print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))

    return mAP, cmc_scores[0]
