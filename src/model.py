import torch
import time
import numpy as np
from random import randint
import os

from utils import *
from scipy import special
import argparse


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


class AE(torch.nn.Module):
    def __init__(self, batch_size, eval_batch_size, device, save_val_dir, save_test_dir):
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device

        self.x = None
        self.height = None
        self.width = None
        self.out_channels = None
        self.kernel_size_cv2d = None
        self.stride_cv2d = None
        self.padding_cv2d = None
        self.kernel_size_mp2d = None
        self.stride_mp2d = None
        self.padding_mp2d = None
        self.alpha = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.save_val_dir = save_val_dir
        self.save_test_dir = save_test_dir

        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        """  Defines a class for an autoencoder algorithm for an object (image) x

        An autoencoder is a specific type of feedforward neural networks where the
        input is the same as the
        output. It compresses the input into a lower-dimensional code and then
        reconstruct the output from this representattion. It is a dimensionality
        reduction algorithm

        Parameters
        ----------
        x : np.array
        a numpy array containing image

        Returns
        ----------
        x-n : np.array
        a numpy array containing the denoised image i.e the image itself minus the noise

        """
        x = torch.permute(x, (0, 3, 1, 2))
        assert x.shape[2] == 256
        assert x.shape[3] == 256
        skips = [x]

        n = x

        # ENCODER
        n = self.leaky(self.enc0(n))
        n = self.leaky(self.enc1(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc2(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc3(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc4(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc5(n))
        n = self.pool(n)
        n = self.leaky(self.enc6(n))

        # DECODER
        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return x - n

    @staticmethod
    def loss_function(output, target):
        """ Defines and runs the loss function

        Parameters
        ----------
        output :
        target :

        Returns
        ----------
        loss: float
            The value of loss given your output, target and batch_size

        """
        loss = torch.mean(torch.square(torch.abs(output-torch.permute(target, (0, 3, 1, 2)))))
        return loss

    @staticmethod
    def generate_speckle(x, L):
        M = 10.089038980848645
        m = -1.429329123112601
        s = torch.zeros(x.size())
        for k in range(L):
            gamma = (torch.abs(torch.complex(torch.normal(0, 1, x.size()), torch.normal(0, 1, x.size()))) ** 2)/2
            s = s+gamma
        s_amplitude = torch.sqrt(s/L)
        s_log = torch.log(s_amplitude)
        log_norm_s = s_log / (M-m)
        return x+log_norm_s

    def training_step(self, batch):
        """ Train the model with the training set

        Parameters
        ----------
        batch : a subset of the training date

        Returns
        -------
        loss : float
          The value of loss given the batch

        """
        L = 1

        x = batch
        y1 = self.generate_speckle(x, L)
        # y2 = self.generate_speckle(x,np.random.randint(20,30))
        # pile = np.concatenate((y1,y2),dim=3)

        x = x.to(self.device)
        y1 = y1.to(self.device)

        out = self.forward(y1)
        loss = self.loss_function(out, x)

        return loss

    def validation_step(self, batch, image_num, epoch_num, eval_files, eval_set):
        """ Test the model with the validation set

        Parameters
        ----------
        batch : a subset of data
        image_num : an ID identifying the feeded image
        epoch_num : an ID identifying the epoch
        eval_files : .npy files used for evaluation in training
        eval_set : directory of dataset used for evaluation in training

        Returns
        ----------
        output_clean_image : a np.array

        """

        x = batch

        L = 1

        y = self.generate_speckle(x, L)
        y = y.to(self.device)
        out = self.forward(y)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        noisyimage = denormalize_sar(np.asarray(y.cpu().numpy()))
        outputimage = denormalize_sar(np.asarray(out.cpu().numpy()))
        groundtruth = denormalize_sar(np.asarray(x.cpu().numpy()))

        # calculate PSNR
        psnr = cal_psnr(outputimage, groundtruth)
        print('img%d PSNR: %.2f' % (image_num, psnr))

        # rename and save
        imagename = eval_files[image_num].split('/')[-1]
        imagename = imagename.replace('.npy', '_L' + str(L) + '_epoch_' + str(epoch_num) + '.npy')

        save_sar_images(outputimage, noisyimage, imagename, self.save_val_dir)

    def test_step(self, im, image_num, test_files, test_set, test_dir):
        pat_size = 256
        stride = 64

        # Pad the image
        # im on gpu ie tensor
        # dimension == [1,1,h,w,1]

        L = 1
        im_gt = denormalize_sar(np.squeeze(np.asarray(im.cpu().numpy())))
        im = self.generate_speckle(im, L)

        im_h, im_w = im.size(dim=1), im.size(dim=2)

        count_image = np.zeros((im_h, im_w))
        out = np.zeros((im_h, im_w))

        if im_h == pat_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0, im_h - pat_size, stride))
            if (x_range[-1] + pat_size) < im_h:
                x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

        if im_w == pat_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, im_w - pat_size, stride))
            if (y_range[-1] + pat_size) < im_w:
                y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))

        # testing by patch
        for x in x_range:
            for y in y_range:
                patch_test = im[:, x:x + pat_size, y:y + pat_size, :]
                patch_test = patch_test.to(self.device)

                tmp = self.forward(patch_test)
                tmp = denormalize_sar(np.asarray(tmp.cpu().numpy()))

                out[x:x + pat_size, y:y + pat_size] = out[x:x + pat_size, y:y + pat_size] + tmp

                count_image[x:x + pat_size, y:y + pat_size] = count_image[x:x +
                                                                          pat_size, y:y + pat_size] + np.ones((pat_size, pat_size))

        out = out/count_image
        # out is de-normalized

        imagename = test_files[image_num].split('/')[-1]

        psnr = cal_psnr(out, im_gt)
        print('img%d PSNR: %.2f' % (image_num, psnr))

        save_sar_images(out, denormalize_sar(np.squeeze(np.asarray(im.cpu().numpy()))),
                        imagename, self.save_test_dir, noisy_bool=False)
