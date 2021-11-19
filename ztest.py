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
x = floor(circles[0][0])
y = floor(circles[0][1])
z = float(Core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)"))

Core.start_continuous_sequence_acquisition(17)
time.sleep(1)
start = time.time()
zs = []
ps = [[],[],[]]
for cot in range(200):
	img = NormalizeImage(Core.get_last_image())
	x, y = SearchNearby(img, x, y, searchRadius, minRadius, maxRadius)
	if (x == 0 or y == 0):
		print("Lost Circle")
		break

	p1, p2, p3 = Phase(img, x, y)
	ps[0].append(p1)
	ps[1].append(p2)
	ps[2].append(p3)
	cz = Core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)")
	zs.append(cz)
	z -= 0.02
	print("z =", cz)
	Core.set_property("MCL NanoDrive Z Stage", "Set position Z (um)", z)

Core.stop_sequence_acquisition()
print("Duration: ", time.time() - start)

plt.scatter(zs, ps[0], label="pos = 1")
plt.scatter(zs, ps[1], label="pos = 2")
plt.scatter(zs, ps[2], label="pos = 3")
plt.legend()
plt.show()