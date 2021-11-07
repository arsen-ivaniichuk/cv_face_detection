from django.shortcuts import render
from .forms import UploadFileForm
import tempfile
import os

from .utils import *
import numpy as np
from .detection.detection import FaceDetectorModuleBuilder
import cv2
import pprint


detector_model_type = "RetinaFace"
detector_model_path = "./retinaface_detection/detection/fd/checkpoints/retinaface/retinaface_mnet025_v1/mnet10"
detector_conf_file = (
    "./retinaface_detection/detection/fd/models/model_conf_detector.yaml"
)
device = "cpu"  # "cuda:0"

facedetector = FaceDetectorModuleBuilder(
    model_type=detector_model_type,
    model_path=detector_model_path,
    conf=detector_conf_file,
    device=device,
)

facedetector_run_args = {
    "save": False,  # unless in debug, use save=False, if True, will save face and sligned image to files final.jpg and aligned.jpg
    "save_dir": "./",
    "align": True,  # weather to align faces, this drastically improves recognition quality
    "scale_detector": 640,  # if you decide to change this parameter, please also change it in config for models that bing scaling to input layer like RetinaFace or BlazeFace
    "scale_recognizer": (
        112,
        112,
    ),  # all embedding models supported currently work with 112x112 face images, please don't change this unless adding support for another model
    "detection_threshold": 0.8,
}  # after running tests to determine correlation between thresh and acc, 0.8 as a detection certainty threshold was selected


def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            results = pprint.pformat(
                handle_uploaded_file(request.FILES["file"]), width=10
            )

            print(results)
            return render(
                request,
                "retinaface_detection/index.html",
                {
                    "form": form,
                    "predicted_label": results,
                },
            )

    else:
        form = UploadFileForm()
    return render(
        request,
        "retinaface_detection/index.html",
        {
            "form": form,
            "predicted_label": "",
        },
    )


def handle_uploaded_file(f):
    extention = f.name.split(".")[1].lower()

    with tempfile.NamedTemporaryFile(
        dir="/tmp/", mode="w+b", suffix="." + extention
    ) as destination:
        print(destination.name)
        for chunk in f.chunks():
            destination.write(chunk)

        if extention == "pdf":
            images = extract_from_pdf(destination.name)
            # print(images)

        elif extention == "docx":
            with tempfile.TemporaryDirectory() as tmpdirname:
                images = extract_from_docx(destination.name, tmpdirname)
                # print(images)

        results = []
        i_faces = 0
        for image in images:
            result = [
                {
                    "x0": int(i["location"][0]),
                    "y0": int(i["location"][1]),
                    "x1": int(i["location"][2]),
                    "y1": int(i["location"][3]),
                    "conf": str(i["location"][4]),
                    "image_width": int(i["location"][2] - i["location"][0]),
                    "image_height": int(i["location"][3] - i["location"][1]),
                }
                for i in facedetector.run(img=image, args=facedetector_run_args)
            ]
            print(result)
            if result:
                results.append({"image_" + str(i_faces): result})

        return {"images_with_faces": results}
