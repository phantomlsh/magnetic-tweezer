import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, QI

# parameters
minRadius = 15
maxRadius = 30
N = 80
Nθ = 80
Nr = 80
sxs, sys = QI.SamplePoints(N, Nθ, Nr)

img = mm.Get()
circles = utils.HoughCircles(img, minRadius, maxRadius)
beads = []
xss = []
n = 0
# find some good circles
for c in circles:
	if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
		beads.append([c[0], c[1]])
		xss.append([])
		n += 1
print(n, "Beads:", beads)

def XYI(img, x, y):
	xl = int(x)
	yl = int(y)
	r = 40
	f = 1.5
	dx = utils.SymmetryCenter(np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0))
	dy = utils.SymmetryCenter(np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1))
	for i in range(2):
		I = utils.BilinearInterpolate(img, sxs+xl+dx, sys+yl+dy)
		I = I.reshape((Nθ, Nr))
		Δx, Δy = QI.XY(I, Nθ)
		dx += Δx * f
		dy += Δy * f
	return xl+dx, yl+dy, I

ts = []
start = time.time()
for loop in range(100):
	img = mm.Get()
	ts.append(time.time() - start)
	for i in range(n):
		beads[i][0], beads[i][1], I = XYI(img, beads[i][0], beads[i][1])
		xss[i].append(beads[i][0])

print('time:', time.time() - start)
print(np.std(xss[0][10:] - np.mean(xss, axis=0)[10:]))
# plt.plot(ts, xss[0] - np.mean(xss, axis=0), marker="o")
# plt.show()
