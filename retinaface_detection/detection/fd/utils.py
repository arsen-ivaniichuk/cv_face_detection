import argparse
import datetime
import glob
import math
import os
import sys
from time import time

import cv2
import numpy as np
import torch
import torchvision.ops.boxes as bops


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"


def draw_faces(faces, landmarks, img, name="", distance=""):
    for i in range(faces.shape[0]):
        box = faces[i].astype(np.int)
        color = (50, 225, 0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            img,
            str(round(faces[i][4], 3)) + " " + name + " " + str(round(distance, 3)),
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    return img


def scale_img(img, scales_original):
    im_shape = img.shape
    target_size = scales_original[0]
    max_size = scales_original[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    return img, im_shape, scales
