import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T as T
import N as N

shape = (700, 756)
img = np.ndarray(shape)
simulate.Generate(50, 50, 50, 4, img)
img0 = np.ndarray(shape)
simulate.Generate(50, 50, 60, 4, img0)

maxn = 20
loopRange = range(1, maxn)

if (True):
    ts = []
    for loop in loopRange:
        beads = []
        for i in range(loop):
            beads.append(T.Bead(50, 50))

        T.XY(beads, img)

        for i in range(20):
            N.Calibrate(beads, [img0], 0)
        for i in range(20):
            N.Calibrate(beads, [img], 2)

        T.ComputeCalibration(beads)

        start = time.time()
        for i in range(100):
            T.XY(beads, img)
            T.Z(beads, img)
        ts.append(time.time() - start)
        beads.append(T.Bead(50, 50))

    plt.plot(loopRange, np.array(ts), marker="o", label="GPU - Taichi")
    print(np.polynomial.polynomial.polyfit(loopRange[1:], ts[1:], 1))

if (True):
    ts = []
    for loop in loopRange:
        beads = []
        for i in range(loop):
            beads.append(N.Bead(50, 50))

        N.XY(beads, img)

        for i in range(20):
            N.Calibrate(beads, [img0], 0)
        for i in range(20):
            N.Calibrate(beads, [img], 2)

        N.ComputeCalibration(beads)

        start = time.time()
        for i in range(100):
            N.XY(beads, img)
            N.Z(beads, img)
        ts.append(time.time() - start)
        beads.append(N.Bead(50, 50))

    plt.plot(loopRange, np.array(ts), marker="o", label="CPU - Numpy")
    print(np.polynomial.polynomial.polyfit(loopRange[1:], ts[1:], 1))

plt.title("Time to run 100 frames")
plt.ylabel("Time(s)")
plt.xlabel("Bead Number")
plt.grid()
plt.legend()
plt.show()