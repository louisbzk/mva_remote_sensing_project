import numpy as np
import cv2
import diplib as dip


def process_line_img(img: [np.ndarray, str],
                     q_lower=1,
                     q_upper=99,
                     contrast_alpha=4.,
                     contrast_beta=0.,
                     thresh_window=11,
                     thresh_C=-8,
                     ao_filter_size=180,
                     return_dtype=np.float32,
                     ) -> np.ndarray:
    """
    Process a line image to remove artifacts, small features... & convert to binary mask

    4 steps :
     - Equalize the image
     - Contrast the image
     - Threshold it
     - Remove small connected features using area opening

    :param img: the input image as an array, or path to input image. It may be RGB or grayscale with 0-255 values.
    :param q_lower: float or None. If a float, will saturate values lower than the q_lower-th percentile
     before processing
    :param q_upper: float or None. If a float, will saturate values higher than the q_lower-th percentile
     before processing
    :param contrast_alpha: the linear ramp used for contrasting. See cv2.convertScaleAbs for details.
    :param contrast_beta: the offset used for contrasting. See cv2.convertScaleAbs for details.
    :param thresh_window: the window size used for thresholding. See cv2.adaptiveThreshold for details.
    :param thresh_C: the offset used for thresholding. See cv2.adaptiveThreshold for details.
    :param ao_filter_size: the filter size used for area opening. See diplib.BinaryAreaOpening for details.
    :param return_dtype: the type of the returned array. Default is float32.
    :return: the processed line image as a binary array (0/1 values of the specified dtype)
    """
    # ensure the image is a 0-255 grayscale image (keep float values for equalization)
    _min, _max = np.min(img), np.max(img)
    if isinstance(img, np.ndarray):
        if _min < 0. or _max > 255.:
            raise ValueError('Input array\'s values do not range from 0 to 255.')
        if img.dtype == np.uint8:
            _img = img
        elif np.isclose(_max, 1., atol=1e-1) and _max <= 1.:
            _img = (255 * img).astype(np.uint8)
        else:
            raise ValueError(f'Input array\'s values are neither 0-255 or 0-1. Found min, max = {_min},  {_max}')

    elif isinstance(img, str):
        _img = cv2.imread(img)
    else:
        raise TypeError(f'Image of wrong type, expected \'np.ndarray\' or \'str\' but got {type(img)}')

    # make sure image is grayscale
    if len(_img.shape) != 2:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)  # RGB, BGR equivalent for grayscale anyway...

    # step 1 : equalize
    if q_lower:
        qmin = np.percentile(_img, q=q_lower)
        _img[_img < qmin] = qmin
    else:
        qmin = np.min(_img)
    if q_upper:
        qmax = np.percentile(_img, q=q_upper)
        _img[_img > qmax] = qmax
    else:
        qmax = np.max(_img)

    _img = (255 * (_img.astype(np.float32) - qmin) / (qmax - qmin)).astype(np.uint8)

    # step 2 : contrast
    _img = cv2.convertScaleAbs(_img, alpha=contrast_alpha, beta=contrast_beta)

    # step 3 : threshold and convert to binary
    _img = cv2.adaptiveThreshold(_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_window, thresh_C)

    # step 4 : remove small connected features by area opening
    _img = dip.BinaryAreaOpening(_img > 0, filterSize=ao_filter_size, connectivity=2)

    return np.array(_img).astype(return_dtype)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    processed_img = process_line_img('../raw_lines/saclay.png')
    # check image is binary
    print(np.all(np.logical_or(processed_img == 0, processed_img == 1)))
    plt.imshow(processed_img, cmap='gray')
    plt.show()
