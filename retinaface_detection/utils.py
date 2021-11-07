import fitz
import docx2txt
import cv2
import numpy as np
import io
from PIL import Image
import os


def extract_from_docx(file_path, tmpdirname):
    print(tmpdirname)
    text = docx2txt.process(file_path, tmpdirname)
    images = [cv2.imread(tmpdirname + "/" + img) for img in os.listdir(tmpdirname)]
    return images


def extract_from_pdf(file_path):

    pdf_file = fitz.open(file_path)
    output = []

    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]

        if page.get_images():
            for image_index, img in enumerate(page.get_images(), start=1):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))

                image = np.array(image)

                output.append(image)

    return output
