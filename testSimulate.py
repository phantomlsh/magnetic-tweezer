import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T
import kernel

π = np.pi

img = np.ndarray((100, 100))
x = 50
y = 50

simulate.Generate(50, 50, 50, 4, img)

beads = []
for i in range(100):
    beads.append(T.Bead(50, 50))
kernel.Profile(beads, img)

start = time.time()

beads = []
for i in range(5):
    beads.append(T.Bead(50, 50))

for i in range(1000):
    kernel.Profile(beads, img)

print("--- %s seconds ---" % (time.time() - start))
