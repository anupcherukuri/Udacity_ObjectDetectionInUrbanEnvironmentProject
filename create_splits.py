import argparse
import glob
import os
import random

import numpy as np
import shutil
from utils import get_module_logger
import tensorflow as tf


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    filenames = [name for name in glob.glob(f'{data_dir}/*.tfrecord')]
    np.random.shuffle(filenames)
    train_set, val_set = np.split(filenames, [int(.85*len(filenames))])
    print(len(train_set))      
    if os.path.exists(os.path.join(data_dir,'train')):
        for file in train_set:
            shutil.move(file, os.path.join(data_dir,'train'))
            print('yesTrain')
    else:
        os.makedirs(os.path.join(data_dir,'train'))
        shutil.move(file, os.path.join(data_dir,'train'))
        
    if os.path.exists(os.path.join(data_dir,'val')):   
            for file in val_set:
                shutil.move(file, os.path.join(data_dir,'val'))
                print('yesVal')
    else:
        os.makedirs(os.path.join(data_dir,'val'))
        shutil.move(file, os.path.join(data_dir,'val'))
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)