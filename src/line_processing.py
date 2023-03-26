import numpy as np
import cv2
import diplib as dip
from PIL import Image, ImageEnhance


def process_line_img(img: [np.ndarray, str],
                     q_lower=1,
                     q_upper=99,
                     contrast_alpha=4.,
                     contrast_beta=0.,
                     thresh_window=11,
                     thresh_C=-8,
                     vignette_contrast_factor=50,
                     preferred_vignette_method=None,
                     ao_filter_size=180,
                     return_dtype=np.float32,
                     ) -> np.ndarray:
    """
    Process a line image to remove artifacts, small features... & convert to binary mask

    6 steps :
     - Convert the image to grayscale
     - Compute the size of the vignette-like artifact of the line detector
     - Contrast the image
     - Threshold it
     - Remove vignette
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
    :param vignette_contrast_factor: contrast factor used for vignette computation. Try higher values if it fails.
    :param preferred_vignette_method: int or None. if int, will prefer the corresponding method in compute_vignette_size
    :param ao_filter_size: the filter size used for area opening. See diplib.BinaryAreaOpening for details.
    :param return_dtype: the type of the returned array. Default is float32.
    :return: the processed line image as a binary array (0/1 values of the specified dtype)
    """
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

    # step 1 : grayscale
    if len(_img.shape) != 2:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)  # RGB, BGR equivalent for grayscale anyway...

    # step 2 : compute vignette
    # a preliminary step is to apply high contrast to the image. This is not the contrast step mentioned in the doc
    pil_img = Image.fromarray(_img, mode='L')
    contraster = ImageEnhance.Contrast(pil_img)
    pil_img = contraster.enhance(factor=vignette_contrast_factor)
    vignette = compute_vignette_size(np.array(pil_img).astype(np.uint8), preferred_method=preferred_vignette_method)

    # step 3 : contrast
    _img = cv2.convertScaleAbs(_img, alpha=contrast_alpha, beta=contrast_beta)

    # step 4 : threshold
    _img = cv2.adaptiveThreshold(_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_window, thresh_C)

    # step 5 : remove vignette
    _img = _apply_vignette_removal(_img, vignette)

    # step 6 : remove small connected features
    _img = dip.BinaryAreaOpening(_img > 0, filterSize=ao_filter_size, connectivity=2)

    return np.array(_img).astype(return_dtype)


def process_lines_soft(img: np.ndarray,
                       q_lower=1,
                       q_upper=99,
                       vignette_contrast_factor=50,
                       preferred_vignette_method=None,
                       return_dtype=np.float32,
                       ):
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

        # step 1 : grayscale
        if len(_img.shape) != 2:
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)  # RGB, BGR equivalent for grayscale anyway...

        pil_img = Image.fromarray(_img, mode='L')
        contraster = ImageEnhance.Contrast(pil_img)
        pil_img = contraster.enhance(factor=vignette_contrast_factor)
        vignette = compute_vignette_size(np.array(pil_img).astype(np.uint8), preferred_method=preferred_vignette_method)

        _img = _apply_vignette_removal(_img, vignette)
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

        _img = ((_img.astype(np.float32) - qmin) / (qmax - qmin)).astype(return_dtype)
        return _img


def compute_vignette_size(img: np.ndarray,
                          prop_px_threshold=0.4,
                          hist_neighborhood_size=4,
                          consistency_thresh=4,
                          preferred_method=None,
                          ):
    """
    Compute the size of the white vignette effect of contrasted line images.

    Three methods:
    1 - iteratively removing one-pixel-wide square vignette as long as at least
    one side of the square has "a lot" of white pixels in it, indicating it is almost
    a perfect white line
    2 - draw a line at each border of the image. For instance, for the bottom border,
    draw a horizontal line at the bottom of the image. Define a "below" and "above" region
    of the line, e.g. 5 pixels below and above. This defines a local rectangle area at
    the border of the image. Move that rectangle up pixel by pixel and compute a histogram
    at either side of the line. When the difference between the histogram is
    maximized, the line should match with the vignette border
    3 - by computing gradients, we identify the border of the vignette as a line
    (NOT IMPLEMENTED)

    once the vignette's border is identified we can easily remove it by padding with zeros.
    :param img: the image to remove the vignette of
    :param prop_px_threshold: proportion of pixel below which the line is considered 'not white' (method 1)
    :param hist_neighborhood_size: size of the neighborhood at each size of the line (method 2)
    :param consistency_thresh: maximum size difference (in pixels) of the vignette estimation by the methods
    :param preferred_method: ignore the consistency check and select the result of one method.
    :return: the un-vignetted image (the vignette is replaced with zeros)
    """
    _im = (img - np.min(img)) / (np.max(img) - np.min(img))
    if preferred_method == 1:
        return _vignette_method_1(_im, prop_px_threshold)
    elif preferred_method == 2:
        return _vignette_method_2(_im, hist_neighborhood_size)
    # method 1
    vignette_1 = _vignette_method_1(_im, prop_px_threshold)  # tuple (left, top)
    # method 2
    vignette_2 = _vignette_method_2(_im, hist_neighborhood_size)
    # check consistency
    if (abs(vignette_2[0] - vignette_1[0]) > consistency_thresh or
            abs(vignette_2[1] - vignette_1[1]) > consistency_thresh):
        raise ValueError(f'Removal of vignette effect failed because the two methods gave inconsistent results. Method '
                         f'1 gave a vignette of shape {vignette_1}, but method 2 gave {vignette_2}')

    vignette = int(0.5 * (vignette_1[0] + vignette_2[0])), int(0.5 * (vignette_1[1] + vignette_2[1]))
    return vignette


def _vignette_method_1(_im, prop_px_threshold):
    h, w = _im.shape
    left, top, right, bot = 0, 0, -1, -1
    while left < w:
        prop_px_left = np.mean(_im[:, left])
        if prop_px_left < prop_px_threshold:
            break
        left += 1

    while top < h:
        prop_px_top = np.mean(_im[top, :])
        if prop_px_top < prop_px_threshold:
            break
        top += 1

    while right > -w:
        prop_px_right = np.mean(_im[:, right])
        if prop_px_right < prop_px_threshold:
            break
        right -= 1

    while bot > -h:
        prop_px_bot = np.mean(_im[bot, :])
        if prop_px_bot < prop_px_threshold:
            break
        bot -= 1

    return max(left+1, -right), max(top+1, -bot)  # number of pixels to remove


def _vignette_method_2(_im, hist_neighborhood_size):
    h, w = _im.shape
    _im_padded = np.pad(_im, pad_width=hist_neighborhood_size, mode='edge')
    left, top = _vignette_method_2_iteration(_im_padded, h, w, hist_neighborhood_size)

    _im_padded = np.flip(_im_padded, axis=(0, 1))
    right, bot = _vignette_method_2_iteration(_im_padded, h, w, hist_neighborhood_size)

    return max(left+1, right+1), max(top+1, bot+1)


def _vignette_method_2_iteration(_im_padded, h, w, hist_neighborhood_size):
    left, top = hist_neighborhood_size, hist_neighborhood_size
    diff_buff = np.zeros(shape=w // 10, dtype=np.float32)
    while left < w // 10:
        sum_before = np.sum(_im_padded[:, left - hist_neighborhood_size:left])
        sum_after = np.sum(_im_padded[:, left:left + hist_neighborhood_size])
        diff_buff[left] = abs(sum_after - sum_before)
        left += 1
    left = np.argmax(diff_buff) - hist_neighborhood_size

    diff_buff = np.zeros(shape=h // 10, dtype=np.float32)
    while top < h // 10:
        sum_before = np.sum(_im_padded[top - hist_neighborhood_size:top, :])
        sum_after = np.sum(_im_padded[top:top + hist_neighborhood_size, :])
        diff_buff[top] = abs(sum_after - sum_before)
        top += 1
    top = np.argmax(diff_buff) - hist_neighborhood_size

    return left, top


def _apply_vignette_removal(img: np.ndarray, vignette_shape, fill_value=0):
    img[:, :vignette_shape[0]] = fill_value
    img[:, -vignette_shape[0]:] = fill_value
    img[:vignette_shape[1], :] = fill_value
    img[-vignette_shape[1]:, :] = fill_value

    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    processed_img = process_line_img('../raw_lines/saclay.png')
    # check image is binary
    print(np.all(np.logical_or(processed_img == 0, processed_img == 1)))
    plt.imshow(processed_img, cmap='gray')
    plt.show()
