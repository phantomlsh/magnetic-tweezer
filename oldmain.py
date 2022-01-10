from init import Core, NormalizeImage

# Config
minRadius = 15
maxRadius = 30
searchRadius = 50

import numpy as np
from matplotlib import pyplot as plt
import time
from math import floor
from position import HoughCircles, SearchNearby, Phase

# Get a rough position
Core.snap_image()
img = NormalizeImage(Core.get_image())
circles = HoughCircles(img, minRadius, maxRadius)
x0 = circles[0][0]
y0 = circles[0][1]
x1 = circles[1][0]
y1 = circles[1][1]

Core.start_continuous_sequence_acquisition(17)
time.sleep(1)
start = time.time()
xs = []
ps = []
ts = []
for i in range(1000):
    img = NormalizeImage(Core.get_last_image())
    x0, y0 = SearchNearby(img, x0, y0, searchRadius, minRadius, maxRadius)
    x1, y1 = SearchNearby(img, x1, y1, searchRadius, minRadius, maxRadius)
    if (x0 == 0 or y0 == 0 or x1 == 0 or y1 == 0):
        print("Lost Circle")
        break
    ps.append(Phase(img, x0, y0) - Phase(img, x1, y1))
    xs.append(x1-x0)
    ts.append(time.time() - start)

Core.stop_sequence_acquisition()
print("Duration: ", time.time() - start)
print("X STD: ", np.std(xs))
print("Phase STD: ", np.std(ps))

fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(ts, xs, label="x")
axs[0].grid()
axs[0].legend()
axs[1].plot(ts, ps, label="phase")
axs[1].grid()
axs[1].legend()
#plt.show()
