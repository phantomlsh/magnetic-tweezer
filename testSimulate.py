import numpy as np
import taichi as ti
import time
import simulate
import matplotlib.pyplot as plt
import T

img = np.ndarray((700, 756))
simulate.Generate(50, 50, 50, 4, img)

beads = []
for i in range(30):
    beads.append(T.Bead(50, 50))

T.test(beads, img)

start = time.time()
for i in range(1000):
    T.test(beads, img)
print("--- %s seconds ---" % (time.time() - start))

print(beads[0].x)
