import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


def process_annotations(root_dir, save_dir):
    """Process all annotation MAT files"""

    annotation_dir = os.path.join(root_dir, 'annotations')
    file_names = sorted(os.listdir(annotation_dir))

    train_imnames = sio.loadmat(os.path.join(root_dir, 'frame_train.mat'))
    train_imnames = train_imnames['img_index_train'].squeeze()
    test_imnames = sio.loadmat(os.path.join(root_dir, 'frame_test.mat'))
    test_imnames = test_imnames['img_index_test'].squeeze()
    train_imnames = [train_name[0] + '.jpg' for train_name in train_imnames]
    test_imnames = [test_name[0] + '.jpg' for test_name in test_imnames]

    train_box_imnames = []
    train_boxes = []
    test_box_imnames = []
    test_boxes = []

    for f_name in tqdm(file_names):
        im_name = f_name[:-4]
        if im_name == '0':
            print('bad image!')
            continue
        f_dir = os.path.join(annotation_dir, f_name)
        boxes = sio.loadmat(f_dir)
        if 'box_new' in boxes.keys():
            boxes = boxes['box_new']
        elif 'anno_file' in boxes.keys():
            boxes = boxes['anno_file']
        elif 'anno_previous' in boxes.keys():
            boxes = boxes['anno_previous']
        else:
            raise KeyError(boxes.keys())
        valid_index = np.where((boxes[:, 3] > 0) & (boxes[:, 4] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(im_name)
        boxes = boxes[valid_index].astype(np.int32)

        if im_name in train_imnames:
            train_boxes.append(boxes)
            train_box_imnames.extend([im_name] * boxes.shape[0])
        elif im_name in test_imnames:
            test_boxes.append(boxes)
            test_box_imnames.extend([im_name] * boxes.shape[0])
        else:
            print('{:s} does not exist'.format(im_name))
            sys.exit()

    # Concat all the boxes
    train_boxes = np.vstack(train_boxes).astype(np.int32)
    test_boxes = np.vstack(test_boxes).astype(np.int32)

    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'cls_id', 'pid']

    train_boxes_df = pd.DataFrame(
        train_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    train_boxes_df['imname'] = train_box_imnames
    train_boxes_df['cls_id'] = np.ones((train_boxes.shape[0], 1),
                                       dtype=np.int32)
    train_boxes_df = train_boxes_df[ordered_columns]

    test_boxes_df = pd.DataFrame(
        test_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    test_boxes_df['imname'] = test_box_imnames
    test_boxes_df['cls_id'] = np.ones((test_boxes.shape[0], 1), dtype=np.int32)
    test_boxes_df = test_boxes_df[ordered_columns]

    train_imnames = pd.Series(train_imnames)
    test_imnames = pd.Series(test_imnames)

    train_test_dfs = (train_boxes_df, test_boxes_df, train_imnames,
                      test_imnames)
    train_test_dfs = fix_train_test(root_dir, train_test_dfs)
    train_test_dfs = remove_outliers(train_test_dfs)

    # Write csv files
    train_boxes_df, test_boxes_df, train_imnames, test_imnames = train_test_dfs
    train_boxes_df.to_csv(os.path.join(save_dir, 'trainAllDF.csv'),
                          index=False)
    test_boxes_df.to_csv(os.path.join(save_dir, 'testAllDF.csv'), index=False)
    train_imnames.to_csv(os.path.join(save_dir, 'trainImnamesSe.csv'), header=None,
                         index=False)
    test_imnames.to_csv(os.path.join(save_dir, 'testImnamesSe.csv'), header=None,
                        index=False)


def fix_train_test(root_dir, train_test_dfs):
    """Fix training set and test set."""

    print('\nFixing training and testing annotation files...')
    train_boxes_df, test_boxes_df, train_imnames, test_imnames = train_test_dfs
    train_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_train.mat'))['ID_train'].squeeze()
    test_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_test.mat'))['ID_test2'].squeeze()

    train_boxes_df.index = range(train_boxes_df.shape[0])
    test_boxes_df.index = range(test_boxes_df.shape[0])

    train_dict = {id_num: i for i, id_num in enumerate(train_ids)}
    test_dict = {id_num: i for i, id_num in enumerate(test_ids)}

    # Change those IDs not in test_ids to -1
    for i in range(test_boxes_df.shape[0]):
        if test_boxes_df.loc[i, 'pid'] not in test_ids:
            test_boxes_df.loc[i, 'pid'] = -1
        else:
            test_boxes_df.loc[i, 'pid'] = test_dict[test_boxes_df.loc[i, 'pid']]

    # Change pid with value -2 in training set to -1
    for i in range(train_boxes_df.shape[0]):
        if train_boxes_df.loc[i, 'pid'] == -2:
            train_boxes_df.loc[i, 'pid'] = -1
        else:
            train_boxes_df.loc[i, 'pid'] = train_dict[
                train_boxes_df.loc[i, 'pid']]

    train_test_dfs = (train_boxes_df, test_boxes_df, train_imnames,
                      test_imnames)

    return train_test_dfs


def remove_outliers(train_test_dfs):
    """Remove the outlier images that contain more than one same person."""

    print('\nRemoving outliers in annotation files')
    train_boxes_df, test_boxes_df, train_imnames, test_imnames = train_test_dfs

    # Test set
    exception_test_indices = []
    for im_name in test_imnames:
        im_df = test_boxes_df[test_boxes_df['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for idx in counter.keys():
            if idx > -1 and counter[idx] > 1:
                pid_df = im_df.query('pid==@idx')
                exc_inds = pid_df.index.tolist()[1:]
                exception_test_indices.extend(exc_inds)
    test_boxes_df.drop(exception_test_indices, axis=0, inplace=True)
    test_boxes_df.index = range(test_boxes_df.shape[0])

    # Training set
    exception_train_indices = []
    for im_name in train_imnames:
        im_df = train_boxes_df[train_boxes_df['imname'] == im_name]
        counter = Counter(im_df['pid'])
        for idx in counter.keys():
            if idx > -1 and counter[idx] > 1:
                pid_df = im_df.query('pid==@idx')
                exc_inds = pid_df.index.tolist()[1:]
                exception_train_indices.extend(exc_inds)
    train_boxes_df.drop(exception_train_indices, axis=0, inplace=True)
    train_boxes_df.index = range(train_boxes_df.shape[0])

    train_test_dfs = (train_boxes_df, test_boxes_df, train_imnames,
                      test_imnames)

    return train_test_dfs


def produce_query_set(root_dir, save_dir):
    """Produce query set"""

    test_ids = sio.loadmat(os.path.join(
        root_dir, 'ID_test.mat'))['ID_test2'].squeeze()
    test_boxes_df = pd.read_csv(os.path.join(save_dir, 'testAllDF.csv'))
    test_dict = {id_num: i for i, id_num in enumerate(test_ids)}

    query_imnames = []
    query_boxes = []

    with open(os.path.join(root_dir, 'query_info.txt')) as f:
        for line in f.readlines():
            line = line.split(' ')
            im_name = line[-1].rstrip() + '.jpg'
            box = np.array([line[:-1]]).astype(np.float32).astype(np.int32)
            query_imnames.append(im_name)
            query_boxes.append(box)

    # Concat all the boxes
    query_boxes = np.vstack(query_boxes)
    # Indicate the order of the column names
    ordered_columns = ['imname', 'x1', 'y1', 'del_x', 'del_y', 'pid']
    query_boxes_df = pd.DataFrame(
        query_boxes, columns=['pid', 'x1', 'y1', 'del_x', 'del_y'])
    query_boxes_df['imname'] = query_imnames
    query_boxes_df = query_boxes_df[ordered_columns]

    for i in range(query_boxes_df.shape[0]):
        query_boxes_df.loc[i, 'pid'] = test_dict[query_boxes_df.loc[i, 'pid']]

    # Add quantity of galleries for every query
    query_num_gt, query_num_gt_cross_camera = [], []
    for i in range(query_boxes_df.shape[0]):
        q_name = query_boxes_df.iloc[i]['imname']
        q_camera = q_name[1]
        pid = query_boxes_df.iloc[i]['pid']
        df = test_boxes_df[test_boxes_df['pid'] == pid]
        num_g = 0

        # `gt_gallery` refers to those images that contain the "pid" person
        gt_gallery = list(set(df['imname']))
        gt_gallery.remove(q_name)
        for gt_im in gt_gallery:
            # Only pick out images under different cameras with query
            if gt_im[1] != q_camera:
                num_g += 1
        query_num_gt.append(len(gt_gallery))
        query_num_gt_cross_camera.append(num_g)

    query_boxes_df['num_gt'] = query_num_gt
    query_boxes_df['num_gt_cross_camera'] = query_num_gt_cross_camera
    query_boxes_df.to_csv(os.path.join(save_dir, 'queryDF.csv'), index=False)


def preprocess(root_dir):
    save_dir = os.path.join(root_dir, 'processed_annotations')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Processing the mat files...')
    process_annotations(root_dir, save_dir)
    print('Producing query files')
    produce_query_set(root_dir, save_dir)


if __name__ == '__main__':
    root_dir = '/data/prw'
    preprocess(root_dir)
