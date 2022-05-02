import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
from utils import BilinearInterpolate
import T
import kernel

π = np.pi

img = np.ndarray((100, 100))
x = 50
y = 50

R = 40
Nθ = 80
Nr = 80
rs = np.tile(np.arange(0, R, R/Nr), Nθ)
θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
# relative sample points
sxs = rs * np.cos(θs)
sys = rs * np.sin(θs)

simulate.Generate(50, 50, 50, 4, img)

Is1 = BilinearInterpolate(img, sxs+x, sys+y)
Is1 = Is1.reshape((Nr, Nθ))
Is1 = np.average(Is1, axis=0)
plt.plot(Is1)

beads = [T.Bead(x, y)]
Is2 = kernel.Profile(img, beads)
plt.plot(Is2[0])
plt.show()

start = time.time()

for i in range(10000):
    Is1 = BilinearInterpolate(img, sxs+x, sys+y)
    Is1 = Is1.reshape((Nr, Nθ))
    Is1 = np.average(Is1, axis=0)

print("--- %s seconds ---" % (time.time() - start))

start = time.time()

beads = []
for i in range(100):
    beads.append(T.Bead(50, 50))

for i in range(10000):
    kernel.Profile(img, beads)

print("--- %s seconds ---" % (time.time() - start))
