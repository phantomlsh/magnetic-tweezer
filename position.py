import cv2 as cv
import numpy as np
from scipy import signal
from math import floor

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
def SymmetryCenter(array):
	l = len(array)
	c = floor(l/2)
	r = floor(l/4)
	normalized = array/(np.max(array)/2) - 1
	co = signal.correlate(normalized, np.flip(normalized), mode="same")
	return (np.argmax(co[c-r:c+r]) - r + c - l/2)/2

def SearchNearby(img, x, y, searchRadius, minRadius, maxRadius):
	region = img[y-searchRadius:y+searchRadius, x-searchRadius:x+searchRadius]
	circles = HoughCircles(region, minRadius, maxRadius)
	if (len(circles) == 0):
		return 0, 0
	x = floor(circles[0][0] + x-searchRadius)
	y = floor(circles[0][1] + y-searchRadius)
	r = floor(circles[0][2]*1.5)
	deltax = SymmetryCenter(np.mean(img[y-1:y+1, x-r:x+r], axis=0))
	deltay = SymmetryCenter(np.mean(img[y-r:y+r, x-1:x+1], axis=1))
	return floor(x+deltax), floor(y+deltay)