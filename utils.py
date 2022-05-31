import numpy as np
from scipy import signal
import cv2 as cv
import matplotlib.pyplot as plt

# Normalize to [-1, 1]
def NormalizeArray(array):
    return 2 * array / np.max(array) - 1

def Gaussian(x, μ, σ):
    return np.exp(-np.power(x-μ, 2.) / (2 * np.power(σ, 2.)))

def CropImage(img, x, y, L):
    xi = int(x)
    yi = int(y)
    r = int(L/2)
    return img[yi-r:yi+r, xi-r:xi+r]

def HoughCircles(image, minRadius, maxRadius):
    img = (image / image.max() * 255).astype("uint8")
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if (circles is None):
        return []
    return circles[0]