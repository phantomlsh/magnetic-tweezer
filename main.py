import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, QI

# parameters
minRadius = 15
maxRadius = 30
N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)


img = mm.Get()
circles = utils.HoughCircles(img, minRadius, maxRadius)
print(circles)
x0 = circles[0][0]
y0 = circles[0][1]
x1 = circles[2][0]
y1 = circles[2][1]

def XYI(img, x, y):
	dx = 0
	dy = 0
	for i in range(3):
		I = utils.BilinearInterpolate(img, xs+x+dx, ys+y+dy)
		I = I.reshape((Nθ, Nr))
		Δx, Δy = QI.XY(I, Nθ)
		dx += Δx * 0.8 - 0.0225
		dy += Δy * 0.8 - 0.0225
	return x+dx, y+dy, I

x0s = []
x1s = []
ds = []
for i in range(100):
	img = mm.Get()
	x0, y0, I = XYI(img, x0, y0)
	x1, y1, I = XYI(img, x1, y1)
	ds.append(x1 - x0)

print(np.std(ds[25:90]))
plt.scatter(range(100), ds)
plt.show()
