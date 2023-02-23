import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Optional, Union
import os
import cv2
import scipy


def line_detection(source_img: Union[str, np.ndarray],
                   crop: Optional[Tuple] = None,
                   max_scale: int = 6,
                   patch_half_size: int = 4,
                   mu: float = 1e-2,  # regularization param for matrix inversion
                   angular_step: float = 1.  # degrees
                   ) -> np.ndarray:
    """
    Line detection algorithm

    :param source_img: black and white image. path of the image, or image as NumPy array
    :param crop: optional tuple for cropping the image, of the form ((top, down), (left, right))
     where each value is the amount of pixels to remove
    :return: the line-detection image produced by the algorithm
    """
    if isinstance(source_img, str):
        im = plt.imread(source_img).astype('double')
    else:
        im = source_img.copy().astype('double')
    if crop:
        im = im[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    im = np.log(im + 1 / 255)
    original_im = im.copy()
    half_profile_size = int(np.ceil(np.sqrt(2) * (patch_half_size + 1)))
    multi_scale_result = np.zeros_like(original_im)
    num_angular_steps = int(np.floor(180 / angular_step))

    # loop for scale from max_scale to 1 (inclusive)
    for scale in range(max_scale, 0, -1):
        newsize = (int(original_im.shape[1] / scale), int(original_im.shape[0] / scale))
        im = cv2.resize(original_im, dsize=newsize, interpolation=cv2.INTER_AREA)
        detection_map = np.full(shape=im.shape, fill_value=np.inf)

        for angle_degrees in np.arange(0., 180., angular_step):
            theta = angle_degrees * np.pi / 180
            xx, yy = np.meshgrid(np.arange(-patch_half_size, patch_half_size + 1),
                                 np.arange(-patch_half_size, patch_half_size + 1))

            # indices of the sparse matrix
            row_indices = np.vstack((
                np.arange(xx.size),
                np.arange(xx.size)
            )).T.flatten()
            col_indices = np.zeros_like(row_indices)

            # values to put in the sparse matrix
            m_values = np.zeros(shape=row_indices.shape, dtype=np.float64)

            k = 0
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    p = np.abs(xx[i, j] * np.cos(theta) + yy[i, j] * np.sin(theta))
                    col_indices[2 * k] = np.floor(p + 1)
                    col_indices[2 * k + 1] = np.ceil(p + 1)
                    m_values[2 * k] = np.abs(p - np.ceil(p))
                    m_values[2 * k + 1] = 1 - m_values[2 * k]  # abs(p - floor(p))

                    k += 1

            # a sparse matrix with value V[k] at index (I[k], J[k])
            M = scipy.sparse.csc_matrix((m_values, (row_indices, col_indices)), shape=(xx.size, half_profile_size))

            # 1 : profile computation
            P = np.linalg.inv(M.T @ M + mu * np.eye(half_profile_size)) @ M.T
            profiles = np.zeros((im.shape[0], im.shape[1], P.shape[0]))

            for i in range(P.shape[0]):
                convol_kernel = P[i, :].reshape((2 * patch_half_size + 1, 2 * patch_half_size + 1))
                profiles[:, :, i] = scipy.signal.convolve2d(im, convol_kernel, mode='same')

            # 2 : forbid values above central profile (at index 0)
            profiles = np.maximum(profiles, profiles[:, :, 0, None])

            # 3 : compute residual energy
            E1 = scipy.signal.convolve2d(im**2, np.ones(2 * patch_half_size + 1)[:, None], mode='same')

            M_full = M.todense()
            E2 = np.zeros_like(E1)
            for i in range(M.shape[1]):
                E2 = E2 - 2 * np.squeeze(profiles[:, :, i]) * scipy.signal.convolve2d(
                    im,
                    M_full[:, i].reshape(2 * patch_half_size + 1, 2 * patch_half_size + 1),
                    mode='same'
                )

            u, s, vh = np.linalg.svd(M_full, full_matrices=False)  # these are np.matrix instances
            u, s, vh = np.asarray(u), np.asarray(s), np.asarray(vh)
            proj = np.zeros_like(profiles)

            for i in range(vh.T.shape[0]):
                proj[:, :, i] = s[i] * np.sum(profiles * vh[i, :][None, :], axis=2)

            E3 = np.sum(proj**2, axis=2)

            E4 = -(scipy.signal.convolve2d(
                im**2, np.ones(2 * patch_half_size + 1)[:, None], mode='same'
            ) - (2 * patch_half_size + 1)**2 * scipy.signal.convolve2d(
                im, np.ones(2 * patch_half_size + 1)[:, None] / (2 * patch_half_size + 1)**2, mode='same'
            )**2
            )

            detection_map = np.minimum(detection_map, E1 + E2 + E3 + E4)

        multi_scale_result = multi_scale_result + cv2.resize(
            detection_map, dsize=(multi_scale_result.shape[1], multi_scale_result.shape[0])
        )

        return multi_scale_result


if __name__ == '__main__':
    test_img = np.load('../../data/train/gt/npy/lely.npy').astype('double')
    line_map = line_detection(test_img)
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.log(1 + test_img), cmap='gray')
    ax[1].imshow(line_map, cmap='gray')
    plt.show()
