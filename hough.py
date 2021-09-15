import cv2 as cv

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
