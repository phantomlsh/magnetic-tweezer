import numpy as np
import taichi as ti
import time
import simulate
import matplotlib.pyplot as plt
import T as T

img = np.ndarray((700, 756))
simulate.Generate(50, 50, 50, 4, img)

beads = []
for i in range(30):
    beads.append(T.Bead(50, 50))

T.XY(beads, img)

T.Calibrate(beads, [img], 0)
T.Calibrate(beads, [img], 1)
T.Calibrate(beads, [img], 4)

T.ComputeCalibration(beads)

start = time.time()
for i in range(1000):
    T.XY(beads, img)
    T.Z(beads, img)
print("--- %s seconds ---" % (time.time() - start))
