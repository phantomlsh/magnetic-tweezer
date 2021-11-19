import cv2 as cv
import numpy as np
from math import floor, atan, sqrt, pi as π
import matplotlib.pyplot as plt
from utils import NormalizeArray, Gaussian
import r2xy

def HoughCircles(image, minRadius, maxRadius):
	circles = cv.HoughCircles(
		image,
		cv.HOUGH_GRADIENT,
		1, 50,
		param1=50,
		param2=30,
		minRadius=minRadius,
		maxRadius=maxRadius
	)
	if (circles is None):
		return []
	return circles[0]

# shift from the center
def SymmetryCenter(array, d):
	l = len(array)
	c = l/2
	normalized = NormalizeArray(array)
	freq = np.fft.rfftfreq(l)
	fft = np.fft.rfft(normalized) * np.exp(2j*π*d*freq)
	co = np.fft.irfft(fft * fft) * Gaussian(np.arange(l), c, c/4)
	#from scipy import signal
	#co = signal.correlate(normalized, np.flip(normalized), mode="same")
	maxi = np.argmax(co)
	p = np.polynomial.polynomial.polyfit(range(maxi-2, maxi+3, 1), co[maxi-2:maxi+3], 2)
	return -p[1]/(4*p[2])-c/2

def SearchNearby(img, cx, cy, searchRadius, minRadius, maxRadius):
	x = floor(cx)
	y = floor(cy)
	region = img[y-searchRadius:y+searchRadius, x-searchRadius:x+searchRadius]
	circles = HoughCircles(region, minRadius, maxRadius)
	if (len(circles) == 0):
		return 0, 0
	x = floor(circles[0][0]) + x-searchRadius
	y = floor(circles[0][1]) + y-searchRadius
	r = floor(circles[0][2]*1.5)
	deltax = SymmetryCenter(np.mean(img[y-2:y+3, x-r:x+r], axis=0))
	deltay = SymmetryCenter(np.mean(img[y-r:y+r, x-2:x+3], axis=1))
	return x+deltax, y+deltay

def Phase(img, cx, cy):
	x = floor(cx)
	y = floor(cy)
	r = 35
	I = np.zeros(r)
	for i in range(10, r, 1):
		ys = np.array(r2xy.Y[i])+y
		xs = np.array(r2xy.X[i])+x
		I[i] = np.sum(np.multiply(img[ys, xs], r2xy.W[i]))

	freq = np.fft.fft(NormalizeArray(I[13:r]))
	freq[0] = 0
	return atan(freq[1].imag/freq[1].real)#, atan(freq[2].imag/freq[2].real), atan(freq[3].imag/freq[3].real)
