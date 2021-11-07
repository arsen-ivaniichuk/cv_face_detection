import cv2
from skimage import transform as trans
import numpy as np
from .transforms import src, src_map


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")

    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    height, width, channel = img.shape
    if channel != 3:
        print("Error input.")
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
