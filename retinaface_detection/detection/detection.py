from __future__ import annotations
from abc import ABC, abstractmethod
from .fd.utils import *
import cv2
import numpy as np
from .fa.utils import norm_crop
from .fd.model_def import (
    ModelLoaderFactory as FaceDetectionModelLoaderFactory,
)


class FaceDetectorModuleBuilder:
    """
    A class used to build a detector, create a model instance and load it to memory,
    detect all faces in the image, first step in a recognition pipeline
    ...

    Attributes
    ----------
    model : ModelClass (ex: RetinaFace)
        Face Detection model instance that is built using model type (ex. RetinaFace), path to weights, hyper parameters and device

    Methods
    -------
    inference(img, scale_to=640, save=True, save_dir="./", detection_threshold=0.8)
        Uses self.model to detect faces in input image

    align(image, landmarks, save=True, save_dir="./")
        alignes face using 5 key facial landmarks

    """

    def __init__(self, model_type, model_path, conf, device):
        """
        Parameters
        ----------

        model_type : str
            type of model Head as in conf file (ex. RetinaFace)
        model_path : str
            path to weights.json
        conf : str
            yaml config file with model hyper parameters
        device : str
            a string representing a device to use for storing models and images, "cuda:<id>" for gpu, "cpu" for cpu

        """
        model_loader_factory = FaceDetectionModelLoaderFactory(
            model_type, model_path, conf
        )
        self.model = model_loader_factory.load_model(device=device)

    def inference(
        self, img, scale_to=640, save=True, save_dir="./", detection_threshold=0.8
    ):
        """
        Parameters
        ----------

        img : numpy.ndarray
            image to detect faces in
        scale_to : int
            scaling parameter for image
        save : bool
            if True save marked faces on image to file
        save_dir : str
            directory to save images
        detection_threshold : float
            detection certainty thresh
        """
        scales_original = [scale_to, scale_to]

        img, im_shape, scales = scale_img(
            img=img,
            scales_original=scales_original,
        )

        inf_start = time()
        faces, landmarks = self.model.detect(
            img, detection_threshold, scales=scales, do_flip=False
        )
        inf_end = time()
        time_diff = inf_end - inf_start
        print("DETECTION TIME: ", time_diff)

        if save:
            self.save(faces, landmarks, img, save_dir)
        return faces, landmarks

    def align(self, image, landmarks, save=True, save_dir="./"):
        """
        Parameters
        ----------

        landmarks : numpy.ndarray
            array of 5 facial key points for each face
        image : numpy.ndarray
            cropped face image to align
        save : bool
            if True save marked faces on image to file
        save_dir : str
            directory to save images
        """
        cropped = norm_crop(image, landmarks)
        if save:
            filename = save_dir + "aligned.jpg"
            cv2.imwrite(filename, cropped)
        return cropped

    def run(
        self,
        img,  # image should be a numpy ndarray as if read by opencv
        args,
    ):
        """
        Parameters
        ----------

        img : numpy ndarray
            image to find faces in
        save : bool
            unless in debug, use save=False, if True, will save face and aligned image to files final.jpg and aligned.jpg
        save_dir : str
            path to save images to
        align : bool
            if True - faces will be aligned using landmarks, this drastically improves face id quality
        scale_detector : int
            default and recommeded for Retina is 640. If you decide to change this parameter,
            please also change it in config for models that bing scaling to input layer like RetinaFace or BlazeFace
        scale_recognizer : tuple
            default (112, 112). All embedding models supported currently work with 112x112 face images, please don't change this unless adding support for another model
        detection_threshold : float
            detection certainty threshold, recommended: 0.8
        """
        # DETECT
        faces, landmarks = self.inference(
            img,
            args["scale_detector"],
            args["save"],
            args["save_dir"],
            args["detection_threshold"],
        )
        aligned_faces = []
        for i in range(len(faces)):
            face = faces[i]
            landmark = np.array(landmarks[i])

            # ALIGN
            aligned_faces.append(
                self.align(
                    img,
                    landmark,
                    save=args["save"],
                    save_dir=args["save_dir"],
                )
            )

        return [
            {
                # "detected_class": "face",
                "location": faces[i],
                # "landmarks": landmarks[i],
            }
            for i in range(len(faces))
        ]
