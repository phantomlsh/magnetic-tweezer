import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T

beads = []

xss = []
circles = utils.HoughCircles(mm.Get(), 15, 30)
for c in circles:
    if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
        beads.append(T.Bead(c[0], c[1]))
        xss.append([])

n = len(beads)
print(n, "Beads:", beads)

T.XY(beads, mm.Get())

ts = []
start = time.time()
for loop in range(1000):
    img = mm.Get()
    ts.append(time.time() - start)
    T.XY(beads, img)
    for i in range(n):
        xss[i].append(beads[i].x)

xss = np.array(xss)
print('time:', time.time() - start)
print('STD:', np.std((xss[0] - xss[1])[10:]))
plt.hist((xss[0] - xss[1])[10:], bins=30)
plt.title('Distribution of Relative X Position Between 2 Ref. Beads')
plt.xlabel('Relative X Position (px)')
plt.show()

