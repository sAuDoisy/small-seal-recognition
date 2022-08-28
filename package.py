# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random

def pickle_examples_with_split_ratio(data_path, save_path, val_ratio=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory

    NOTICE!!! val_ratio here is not "split", just choose some picture randomly to test the picture generate effect
    all pic in data_path will be packaged!
    """
    train_path = os.path.join(save_path, 'train.obj')
    val_path = os.path.join(save_path, 'val.obj')
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            list_dir = os.listdir(data_path)
            # label = int(label)
            for each_pic in list_dir:
                label = each_pic[:-4]
                with open(os.path.join(data_path, each_pic), 'rb') as f:
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < val_ratio:
                        pickle.dump(example, fv)
                    pickle.dump(example, ft)
                    print('img', label, 'saved')


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--data_dir', required=True, help='path of examples')
parser.add_argument('--save_dir', required=True, help='path to save pickled files')
parser.add_argument('--val_ratio', type=float, default=0.1, dest='split_ratio',
                    help='split ratio between train and val')

args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    pickle_examples_with_split_ratio(args.data_dir, args.save_dir, args.val_ratio)