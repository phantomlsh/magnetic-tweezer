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

def PlotCalibration(bead):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.set_title("Profile")
    ax0.imshow(np.flip(bead.Ic, axis=0), cmap="gray")
    ax1.set_title("Real Part")
    ax1.imshow(np.flip(bead.Rc, axis=0), cmap="gray")
    ax2.set_title("Phase")
    ax2.imshow(np.flip(bead.Φc, axis=0))
    plt.show()