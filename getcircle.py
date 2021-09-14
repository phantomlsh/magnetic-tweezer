import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def _show(name, image):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def calcGrayHist(image):
    rows,clos = image.shape
    # grey matrix
    grahHist = np.zeros([256],np.uint64)
    print('initial matrix')
    print(grahHist )
    for r in range(rows):
        for c in range(clos):
            # put grey value
            grahHist[image[r][c]] +=1
    print('valued matrix')
    print(grahHist)
    return grahHist

def hist(image):
    rows, cols = image.shape
    # calculate grey histogram
    grayHist = calcGrayHist(image)
    # cumulative grey histogram
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # mapping input and output
    output = np.zeros([256], np.uint8)
    cofficient = 256.0 / (rows * cols)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            output[p] = np.math.floor(q)
        else:
            output[p] = 0
    # get the balanced image
    equalHistimg = np.zeros(image.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistimg[r][c] = output[image[r][c]]

    return equalHistimg

def scaling(image):
    out = 1.5 * image
    # cutting ceiling
    out[out > 255] = 255
    # type change
    out = np.around(out)
    out = out.astype(np.uint8)
    return out

img = cv2.imread('./resources/calimages/210.901.bmp')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#median_img = cv2.blur(gray_img, (3, 3))
img_bilater = cv2.bilateralFilter(gray_img, 3, 100, 20)

# _, thresh0 = cv2.threshold(scaling(gray_img), 50, 255, cv2.THRESH_BINARY)
#_show("0", img)
#_show("1", median_img)

circles = cv2.HoughCircles(img_bilater, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=30, minRadius=2, maxRadius=20)

circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
    # print(i[0], i[1])
    # draw the outer circle
    # cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    # w = 2 * i[2]

# cv2.imshow("circle" + str(i[0]) + ", " + str(i[1]), img[i[1]-w:i[1]+w, i[0]-w:i[0]+w])
# cv2.waitKey(0)

# [470, 1136]
X = 1136
Y = 470
w = 50

yarr = img[Y-w:Y+w, X, 1]
xarr = img[Y, X-w:X+w, 1]
co = np.fft.ifft(np.fft.fft(xarr)*np.fft.fft(np.flip(xarr)))

X = 1137
Y = 471
w = 50

I = np.zeros(w)
for x in range(-w, w, 1):
  for y in range(-w, w, 1):
    r = math.sqrt(x**2 + y**2)
    if r + 1 >= w:
      continue
    f = math.floor(r)
    I[f] += img[Y+y, X+x, 1]*(1+f-r)
    I[f+1] += img[X+x, Y+y, 1]*(r-f)

for r in range(1, w, 1):
  I[r] /= r

plt.plot(I)
I = I[10:w]
freq = np.fft.fft(I)
freq[0] = 0
#plt.plot(np.abs(freq))
#plt.plot(np.arctan(np.imag(freq)/np.real(freq)))
plt.show()
