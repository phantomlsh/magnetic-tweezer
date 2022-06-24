import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T as _T
import N as N

img = np.ndarray((700, 756))
simulate.Generate(50, 50, 50, 4, img)

maxn = 20

def run():
    ts = []
    for loop in range(1, maxn):
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

    plt.plot(ts)
    print(np.polynomial.polynomial.polyfit(range(1, maxn), ts, 1))

T = _T
run()
T = N
run()

plt.show()