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
xss2 = []
n = 0
for c in circles:
	if (c[0] > 80 and c[0] < 500 and c[1] > 80 and c[1] < 680):
		beads.append([c[0], c[1]])
		xss.append([])
		xss2.append([])
		n += 1
print(n, "Beads:", beads)

def XYI2(img, x, y):
	xl = int(x)
	yl = int(y)
	r = 50
	dx = utils.SymmetryCenter(np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0))
	dy = utils.SymmetryCenter(np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1))
	return x+dx, y+dy

its = []
def XYI(img, x, y):
	xl = int(x)
	yl = int(y)
	r = 50
	dx = utils.SymmetryCenter(np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0))
	dy = utils.SymmetryCenter(np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1))
	it = 9
	for i in range(10):
		I = utils.BilinearInterpolate(img, sxs+x+dx, sys+y+dy)
		I = I.reshape((Nθ, Nr))
		Δx, Δy = QI.XY(I, Nθ)
		dx += Δx * 0.8
		dy += Δy * 0.8
		if (Δx**2 + Δy**2 < 0.001):
			it = i
			break
	its.append(it)
	return x+dx, y+dy, I

ts = []
start = time.time()
for loop in range(100):
	img = mm.Get()
	ts.append(time.time() - start)
	for i in range(n):
		b = beads[i]
		b[0], b[1], I = XYI(img, b[0], b[1])
		xss[i].append(b[0])
		x2, y2 = XYI2(img, b[0], b[1])
		xss2[i].append(x2)

print("Mean Iteration =", np.mean(its))
print(np.std(xss[0][10:] - np.mean(xss, axis=0)[10:]))
plt.plot(ts, xss[0] - np.mean(xss, axis=0), marker="o")
print(np.std(xss2[0][10:] - np.mean(xss2, axis=0)[10:]))
plt.plot(ts, xss2[0] - np.mean(xss2, axis=0), marker="o")
plt.show()
