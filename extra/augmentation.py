#Imports
import numpy as np
import matplotlib.pyplot as plt

import glob
import random

import torchvision.transforms as transforms
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance
import cv2
import skimage.exposure

from PIL import Image
import glob

def convertImage(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")

    datas = img.getdata()

    newData = []

    for item in datas:
        rbg_min = 250
        if item[0] >= rbg_min and item[1] >= rbg_min and item[2] >= rbg_min:
            newData.append((item[0], item[1], item[2], 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save("./New.png", "PNG")
    print("Successful")


def convert2(image_path):
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow('gray_blur', blur)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Obtain bounding rectangle and extract ROI
    x, y, w, h = cv2.boundingRect(thresh)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    ROI = original[y:y + h, x:x + w]

    # Add alpha channel
    b, g, r = cv2.split(ROI)
    alpha = np.ones(b.shape, dtype=b.dtype) * 50
    ROI = cv2.merge([b, g, r, alpha])

    cv2.imshow('thresh', thresh)
    cv2.imshow('image', image)
    cv2.imshow('ROI', ROI)
    cv2.waitKey()


image_list = glob.glob('images_to_test/*')
# convertImage(image_list[5])
convert2(image_list[5])
