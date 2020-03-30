import numpy as np
import cv2 as cv
import os

def kspace_to_image(kspace):
    assert(len(kspace.shape) == 3 and kspace.shape[-1] == 2)
    fft = cv.idft(kspace, flags=cv.DFT_SCALE)
    img = cv.magnitude(fft[:,:,0], fft[:,:,1])
    return cv.normalize(img, dst=None, alpha=255, beta=0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


data = np.load(os.path.join('data', 'e14155s3_P69120.7.npy'))
num_slices, height, width = data.shape[0:3]
data /= np.sqrt(height * width)


WIN_NAME = 'OpenVINO super resolution'

def callback(slice_id):
    kspace = data[slice_id]
    img = kspace_to_image(kspace)

    h, w = img.shape[0], img.shape[1]
    img = cv.resize(img, (w * 3, h * 3), interpolation=cv.INTER_CUBIC)

    cv.imshow(WIN_NAME, img)
    cv.waitKey(1)

cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
cv.createTrackbar('Slice', WIN_NAME, num_slices // 2, num_slices - 1, callback)
callback(num_slices // 2)  # Trigger initial visualization
cv.waitKey()
