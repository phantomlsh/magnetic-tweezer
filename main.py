import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, QI
import T

# parameters
minRadius = 15
maxRadius = 30
QI.Init(80, 80, 80, 1)

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

ts = []
start = time.time()
for loop in range(500):
	img = mm.Get()
	ts.append(time.time() - start)
	for i in range(n):
		beads[i][0], beads[i][1] = T.XY(img, beads[i][0], beads[i][1])
		xss[i].append(beads[i][0])

print('time:', time.time() - start)
print('STD:', np.std(xss[0][10:] - np.mean(xss[1:], axis=0)[10:]))
plt.plot(ts[10:], (xss[0] - np.mean(xss[1:], axis=0))[10:], marker="o")
plt.title('Relative X of 1 bead w.r.t. mean X of beads')
plt.xlabel('Time')
plt.show()

