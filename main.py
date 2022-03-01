import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, QI

# parameters
minRadius = 15
maxRadius = 30
QI.Init(80, 80, 80)

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
		QI.Interpolate(img, xl+dx, yl+dy)
		Δx, Δy = QI.XY()
		dx += Δx * f
		dy += Δy * f
	return xl+dx, yl+dy, QI.Profile()

ts = []
start = time.time()
for loop in range(100):
	img = mm.Get()
	ts.append(time.time() - start)
	for i in range(n):
		beads[i][0], beads[i][1], I = XYI(img, beads[i][0], beads[i][1])
		xss[i].append(beads[i][0])

print('time:', time.time() - start)
print('STD:', np.std(xss[0][10:] - np.mean(xss, axis=0)[10:]))
plt.plot(ts[10:], (xss[0] - np.mean(xss, axis=0))[10:], marker="o")
plt.title('Relative X of 1 bead w.r.t. mean X of 6 beads')
plt.xlabel('Time')
plt.show()
