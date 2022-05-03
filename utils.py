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

# shift from the center
def SymmetryCenter(array, it=1):
    l = len(array)
    r = int(l/2)
    normalized = NormalizeArray(array)
    freq = np.fft.rfftfreq(l*2)
    fft = np.fft.rfft(np.append(normalized, np.zeros(l)))
    res = 0
    d = 0
    for loop in range(it):
        fft *= np.exp(2j*np.pi*freq*d)
        co = np.fft.irfft(fft * fft)[r-1:l+r-1]
        i = np.argmax(co[r-30:r+30]) + r-30
        p = np.polynomial.polynomial.polyfit(range(i-2, i+3), co[i-2:i+3], 2)
        d = -p[1]/(4*p[2])-r/2
        res += d
    return res

def Tilde(I, rf, w):
    I = np.append(np.flip(I), I)
    Iq = np.fft.fft(I)
    q = np.fft.fftfreq(I.shape[-1])
    L = len(Iq)
    win = np.append(np.zeros(w[0]), np.hanning(w[1]-w[0]))
    win = np.append(win, np.zeros(L-w[1]))
    It = np.fft.ifft(Iq*win)
    return It[rf+(len(It)//2):len(It)]

def HoughCircles(image, minRadius, maxRadius):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if (circles is None):
        return []
    return circles[0]