import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T as T
import N as N

img = np.ndarray((700, 756))
simulate.Generate(50, 50, 50, 4, img)

maxn = 20

ts = []
for loop in range(2, maxn):
    beads = []
    for i in range(loop):
        beads.append(T.Bead(50, 50))

    T.XY(beads, img)
    # plt.plot(beads[0].profile)
    # plt.show()

    for i in range(5):
        T.Calibrate(beads, [img], 2)

    T.ComputeCalibration(beads)

    start = time.time()
    for i in range(100):
        T.XY(beads, img)
        T.Z(beads, img)
    ts.append(time.time() - start)
    beads.append(T.Bead(50, 50))

plt.plot(100 / np.array(ts))
print(np.polynomial.polynomial.polyfit(range(2, maxn), ts, 1))

ts = []
for loop in range(2, maxn):
    beads = []
    for i in range(loop):
        beads.append(N.Bead(50, 50))

    N.XY(beads, img)
    # plt.plot(beads[0].profile)
    # plt.show()

    for i in range(5):
        N.Calibrate(beads, [img], 2)

    N.ComputeCalibration(beads)

    start = time.time()
    for i in range(100):
        N.XY(beads, img)
        N.Z(beads, img)
    ts.append(time.time() - start)
    beads.append(N.Bead(50, 50))

plt.plot(100 / np.array(ts))
print(np.polynomial.polynomial.polyfit(range(2, maxn), ts, 1))

plt.show()