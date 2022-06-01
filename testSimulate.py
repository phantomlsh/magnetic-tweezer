import numpy as np
import taichi as ti
import time
import simulate
import matplotlib.pyplot as plt
import T
import kernel

img = np.ndarray((789, 756))
kernel.SetParams(789, 756)
simulate.Generate(50, 50, 50, 4, img)

beads = []
for i in range(5):
    beads.append(T.Bead(50, 50))
kernel.Profile(beads, img)

start = time.time()
for i in range(1000):
    kernel.Profile(beads, img)
print("--- %s seconds ---" % (time.time() - start))
