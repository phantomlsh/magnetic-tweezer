import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T as T
import N as N

shape = (700, 756)
imgc = []
for i in range(20):
    img = np.ndarray(shape)
    simulate.Generate(50, 50, 50 + i, 4, img)
    imgc.append(img)
img = np.ndarray(shape)
simulate.Generate(50, 50, 60, 4, img)

maxn = 20
loopRange = range(1, maxn)

if (True):
    ts = []
    for loop in loopRange:
        beads = []
        for i in range(loop):
            beads.append(T.Bead(50, 50))

        T.XY(beads, [img])

        for i in range(20):
            T.Calibrate(beads, [imgc[i]], i)

        T.ComputeCalibration(beads)
        print(loop, beads[0])

        start = time.time()
        for i in range(100):
            T.XYZ(beads, [img])
        ts.append(time.time() - start)

    print(ts)
    plt.plot(loopRange, np.array(ts), marker="o", label="GPU - Taichi")
    print(np.polynomial.polynomial.polyfit(loopRange[1:], ts[1:], 1))

if (True):
    ts = []
    for loop in loopRange:
        beads = []
        for i in range(loop):
            beads.append(N.Bead(50, 50))

        N.XY(beads, [img])

        for i in range(20):
            N.Calibrate(beads, [imgc[i]], i)

        N.ComputeCalibration(beads)
        N.XYZ(beads, [img])
        print(loop, beads[0])

        start = time.time()
        for i in range(100):
            N.XYZ(beads, [img])
        ts.append(time.time() - start)

    print(ts)
    plt.plot(loopRange, np.array(ts), marker="o", label="CPU - Numpy")
    print(np.polynomial.polynomial.polyfit(loopRange[1:], ts[1:], 1))

plt.title("Time to run 100 frames")
plt.ylabel("Time(s)")
plt.xlabel("Bead Number")
plt.grid()
plt.legend()
plt.show()