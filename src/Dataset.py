import numpy as np
import torch
from utils import *
from torch.utils.data import Dataset
from pathlib import Path


class CustomDataset(Dataset):
    'characterizes a dataset for pytorch'

    def __init__(self, patche):
        self.patches = patche

        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        'denotes the total number of samples'
        return len(self.patches)

    def __getitem__(self, index):
        'Generates one sample of data'
        # select sample
        batch_clean = (self.patches[index, :, :, :])
        x = torch.tensor(batch_clean)

        return x


class ValDataset(Dataset):
    'characterizes a dataset for pytorch'

    def __init__(self, test_set, line_detection_path):
        self.files = glob(test_set+'/*.npy')
        if line_detection_path:
            self.line_files = glob(line_detection_path + '/*.npy')
            _lines_filenames = [Path(f).stem for f in self.line_files]
            # match files one to one
            if len(self.files) != len(self.line_files):
                raise ValueError(f'Wrong number of line detection images : {len(self.line_files)} were found, but '
                                 f'there are {len(self.files)} testing images')
            for i, path in enumerate(self.files):
                try:
                    line_idx = _lines_filenames.index(Path(path).stem)
                except ValueError:
                    raise ValueError(f'Could not find line image that matches the following image : \'{path}\' in '
                                     f'directory \'{line_detection_path}\'')
                if i != line_idx:  # move element in list, so that indices match between the two lists
                    line_path = self.line_files.pop(line_idx)
                    self.line_files.insert(i, line_path)
        else:
            self.line_files = None

        self.data = []

        raw_data = load_sar_images(self.files)
        raw_data = [normalize_sar(im) for im in raw_data]
        n_raw_channels = raw_data[0].shape[-1]
        for i in range(len(self.files)):
            if self.line_files:
                self.data.append(np.zeros(shape=(*raw_data[i].shape[:-1], n_raw_channels + 1), dtype=np.float32))
                line_img = np.load(self.line_files[i])
                if raw_data[i].shape[:-1] != line_img.shape:
                    raise ValueError(f'Size mismatch between raw image and line image in testing set. '
                                     f'Image \'{self.files[i]}\' has size {raw_data[i].shape[:-1]} '
                                     f'(and {n_raw_channels} channels), but line image'
                                     f' \'{self.line_files[i]}\' has shape {line_img.shape}')
                self.data[-1][:, :, -1] = line_img
            else:
                self.data.append(np.zeros(shape=(*raw_data[i].shape, n_raw_channels), dtype=np.float32))
            self.data[-1][:, :, :n_raw_channels] = raw_data[i]

    def __len__(self):
        'denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # select sample
        return torch.from_numpy(self.data[index]).type(torch.float)
