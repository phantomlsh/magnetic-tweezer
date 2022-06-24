import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T

beads = []

circles = utils.HoughCircles(mm.Get(), 15, 30)
for c in circles:
    if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
        beads.append(T.Bead(c[0], c[1]))

n = len(beads)
print(n, "Beads:", beads)

#T.test(beads, mm.Get())

x = np.arange(-2, 3)
y = 2 * x ** 2
def tiFitCenter(y0, y1, y2, y3, y4):
    a = (2*y0 - y1 - 2*y2 - y3 + 2*y4) / 14
    b = -0.2*y0 - 0.1*y1 + 0.1*y3 + 0.2*y4
    return -b/a/2
print(y)
print(tiFitCenter(y[0], y[1], y[2], y[3], y[4]))