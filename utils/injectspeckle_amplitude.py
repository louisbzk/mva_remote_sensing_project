#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:12:23 2018

@author: emasasso
"""
import numpy as np
import os
import matplotlib.pyplot as plt


def injectspeckle_amplitude(img, L):
    rows = img.shape[0]
    columns = img.shape[1]
    s = np.zeros((rows, columns))
    for k in range(0, L):
        gamma = np.abs(np.random.randn(rows, columns) + np.random.randn(rows, columns)*1j)**2/2
        s = s + gamma
    s_amplitude = np.sqrt(s/L)
    ima_speckle_amplitude = np.multiply(img, s_amplitude)
    return ima_speckle_amplitude


if __name__ == '__main__':
    clean_imgs_dir = '../data/train/gt/npy'
    noised_imgs_dir = '../data/train/raw/npy'
    # for root, dirs, files in os.walk(clean_imgs_dir):
    #     for filename in files:
    #         im = np.load(os.path.join(root, filename)).astype('double')
    #         speckled_im = injectspeckle_amplitude(im, 1)
    #         np.save(os.path.join(os.path.join(noised_imgs_dir, filename)), speckled_im)
    im = np.load(os.path.join(noised_imgs_dir, 'lely.npy'))
    plt.imshow(np.log(im + 1/255), cmap='gray')
    plt.show()
