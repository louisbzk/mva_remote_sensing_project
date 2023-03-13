#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:56:07 2018

@author: emasasso
"""

from PIL import Image
import numpy as np
from glob import glob

working_dir = './Test/'
test_files = glob(working_dir + '*.npy')

choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22, 'risoul': 306.94,
           'limagne': 178.43}
for filename in test_files:
    dim = np.load(filename)
    dim = np.squeeze(dim)
    for x in choices:
        if x in filename:
            threshold = choices.get(x)
        else:
            threshold = np.mean(dim) + 3 * np.std(dim)

    dim = np.clip(dim, 0, threshold)
    dim = dim / threshold * 255
    dim = Image.fromarray(dim.astype('float64')).convert('L')
    imagename = filename.replace('npy', 'png')
    dim.save(imagename)
