import glob
import random
import os
import numpy as np
from scipy import signal
from scipy import special
from pathlib import Path


'''
Generate patches for the images in the folder dataset/data/Train
The code scans among the training images and then for data_aug_times
'''


class GenerateDataset:
    @staticmethod  # TODO : change paths
    def generate_patches(src_dir='data/train/raw/npy',
                         lines_dir=None,
                         pat_size=256,
                         step=0,
                         stride=64,
                         bat_size=4,
                         data_aug_times=1,
                         n_channels=1
                         ):
        count = 0
        filepaths = glob.glob(src_dir + '/*.npy')
        if lines_dir:
            lines_filepaths = glob.glob(lines_dir + '/*.npy')
            _lines_filenames = [Path(fp).stem for fp in lines_filepaths]
            # check one-to-one mapping
            if len(filepaths) != len(lines_filepaths):
                raise ValueError(f'Wrong number of line detection images : {len(lines_filepaths)} were found, but '
                                 f'there are {len(filepaths)} training images')
            for i, path in enumerate(filepaths):
                try:
                    line_idx = _lines_filenames.index(Path(path).stem)
                except ValueError:
                    raise ValueError(f'Could not find line image that matches the following image : \'{path}\' in '
                                     f'directory \'{lines_dir}\'')
                if i != line_idx:  # move element in list, so that indices match between the two lists
                    line_path = lines_filepaths.pop(line_idx)
                    lines_filepaths.insert(i, line_path)

        print('number of training data %d' % len(filepaths))

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])

            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count * data_aug_times

        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print('total patches = %d , batch size = %d, total batches = %d' %
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches = int(numPatches)
        inputs = np.zeros((numPatches, pat_size, pat_size, n_channels), dtype='float32')

        count = 0
        # generate patches
        if lines_dir:
            _n_channels_raw_imgs = n_channels - 1
        else:
            _n_channels_raw_imgs = n_channels
        for i in range(len(filepaths)):  # scan through images
            img = (np.load(filepaths[i]))
            img_s = img.reshape((img.shape[0], img.shape[1], _n_channels_raw_imgs))
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)

            if lines_dir:
                line_img = np.load(lines_filepaths[i])
                line_im_h, line_im_w = line_img.shape
                if line_im_h != im_h or line_im_w != im_w:
                    raise ValueError(f'For image \'{filepaths[i]}\', corresponding line image \'{lines_filepaths[i]}\' '
                                     f'size does not match : {(im_h, im_w)} versus {line_img.shape}')

            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    inputs[count, :, :, 0] = img_s[x:x + pat_size, y:y + pat_size, 0]
                    if lines_dir:
                        inputs[count, :, :, 1] = line_img[x:x + pat_size, y:y + pat_size]

                    count += 1

        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

        return inputs
