from init import Core, NormalizeImage

# Config
minRadius = 15
maxRadius = 25
searchRadius = 50

import numpy as np
from matplotlib import pyplot as plt
import time
from math import floor, sqrt
from position import HoughCircles, SymmetryCenter

# Get a rough position
Core.snap_image()
img = NormalizeImage(Core.get_image())
circles = HoughCircles(img, minRadius, maxRadius)
x = floor(circles[0][0])
y = floor(circles[0][1])
z = float(Core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)"))

Core.start_continuous_sequence_acquisition(17)
time.sleep(1)
for cot in range(10):
    img = NormalizeImage(Core.get_last_image())
    region = img[y-searchRadius:y+searchRadius, x-searchRadius:x+searchRadius]
    circles = HoughCircles(region, minRadius, maxRadius)
    if (len(circles) == 0):
        print("Lost circles")
        break
    x = floor(circles[0][0] + x-searchRadius)
    y = floor(circles[0][1] + y-searchRadius)
    r = floor(circles[0][2]*1.5)
    deltax = SymmetryCenter(img[y, x-r:x+r])
    deltay = SymmetryCenter(img[y-r:y+r, x])
    x = floor(x+deltax)
    y = floor(y+deltay)

    r = floor(r * 1.5)
    I = np.zeros(r)
    for i in range(-r, r, 1):
      for j in range(-r, r, 1):
        d = sqrt(i**2 + j**2)
        if d + 1 >= r:
          continue
        f = floor(d)
        I[f] += img[y+i, x+j]*(1+f-d)
        I[f+1] += img[y+i, x+j]*(d-f)

    for i in range(1, r, 1):
        I[i] /= i
    print(Core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)"))
    plt.plot(I)
    z -= 0.5
    Core.set_property("MCL NanoDrive Z Stage", "Set position Z (um)", z)

plt.show()
Core.stop_sequence_acquisition()
