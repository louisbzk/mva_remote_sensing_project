import gc
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob
from GenerateDataset import GenerateDataset

basedir = '.'

# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


class TrainData:
    def __init__(self, filepath='%s/data/image_clean_pat.npy' % basedir):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print('[!] Data file not exists')
            sys.exit(1)

    def __enter__(self):
        print('[*] Loading data...')
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print('[*] Load successfully...')
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print('In __exit__()')


def load_data(filepath='%s/data/image_clean_pat.npy' % basedir):
    return TrainData(filepath=filepath)


def normalize_sar(im):
    return ((np.log(im+1e-6)-m)/(M-m)).astype(np.float32)


def denormalize_sar(im):
    return np.exp((np.clip(np.squeeze(im), 0, 1))*(M-m)+m)


# TODO: add control on training data: exit if does not exists
def load_train_data(filepath, patch_size, batch_size, stride_size, n_data_augmentation, line_detection_path):
    datagen = GenerateDataset()
    n_channels = 2 if line_detection_path else 1
    imgs = datagen.generate_patches(src_dir=filepath, lines_dir=line_detection_path, pat_size=patch_size, step=0,
                                    stride=stride_size, bat_size=batch_size, data_aug_times=n_data_augmentation,
                                    n_channels=n_channels)

    imgs[:, :, :, 0] = normalize_sar(imgs[:, :, :, 0])
    return imgs


def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        return np.array(im).reshape(np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        data.append(np.array(im).reshape(np.size(im, 0), np.size(im, 1), 1))
    return data


def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy', 'png'))


def save_sar_images(denoised, noisy, imagename, save_dir, noisy_bool=True, groundtruth=None):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'domancy': 560, 'Brazil': 103.0,
               'region1_HH': 1068.85, 'region1_HV': 402.75, 'region2_HH': 1039.59, 'region2_HV': 391.70}

    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(noisy) + 3 * np.std(noisy)

    if groundtruth:
        groundtruthfilename = os.path.join(save_dir, 'groundtruth_' + imagename)
        np.save(groundtruthfilename, groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = os.path.join(save_dir, 'denoised_' + imagename)
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    if noisy_bool:
        noisyfilename = os.path.join(save_dir, 'noisy_' + imagename)
        np.save(noisyfilename, noisy)
        store_data_and_plot(noisy, threshold, noisyfilename)


def save_real_imag_images(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(imag_part) + 3 * np.std(imag_part)

    realfilename = os.path.join(save_dir, '/denoised_real_', imagename)
    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = os.path.join(save_dir, '/denoised_imag_', imagename)
    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)


def save_real_imag_images_noisy(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(np.abs(imag_part)) + 3 * np.std(np.abs(imag_part))

    realfilename = save_dir + '/noisy_real_' + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2)*np.abs(real_part), threshold, realfilename)

    imagfilename = save_dir + '/noisy_imag_' + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2)*np.abs(imag_part), threshold, imagfilename)


def save_residual(filepath, residual):
    residual_image = np.squeeze(residual)
    plt.imsave(filepath, residual_image)


def cal_psnr(Shat, S):
    # takes amplitudes in input
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=2.0)
        print('[*] inizialized weights')


def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {'loss': loss.item(), 'accuracy': accuracy.item()}


def save_model(model, destination_folder):
    """
      save the ".pth" model in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(destination_folder)
        print('The new directory is created!')

        torch.save(model.state_dict(), destination_folder+'/model.pth')

    else:
        torch.save(model.state_dict(), destination_folder+'/model.pth')


def save_checkpoint(model, destination_folder, epoch_num, optimizer, loss):
    """
      save the ".pth" checkpoint in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:
        os.makedirs(destination_folder)

    torch.save({
        'epoch_num': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, destination_folder+'/checkpoint_'+str(epoch_num)+'.pth')
    print('\n Checkpoint saved at :', destination_folder)
