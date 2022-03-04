import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, QI, Z

# parameters
minRadius = 15
maxRadius = 30
it = 2
QI.Init(80, 80, 80, 1)
Z.Init(80, 20)

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
	dx = utils.SymmetryCenter(np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0))
	dy = utils.SymmetryCenter(np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1))
	for i in range(it):
		QI.Interpolate(img, xl+dx, yl+dy)
		Δx, Δy = QI.XY()
		dx += Δx
		dy += Δy
	return xl+dx, yl+dy, QI.Profile()


img = mm.Get()
x, y, I = XYI(img, 225, 260)
It = Z.Tilde(I)
plt.plot(It)
plt.show()
