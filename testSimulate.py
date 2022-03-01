import numpy as np
import time
import simulate
from utils import BilinearInterpolate, NormalizeArray
from scipy import interpolate

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
sxs = np.repeat([0], 10000)
sys = np.repeat([0], 10000)
xs = np.tile(np.arange(-50, 50), 100)
ys = np.repeat(np.arange(-50, 50), 100)

simulate.Generate(50+0.1, 50, 50, 4, img)

start = time.time()
for i in range(10):
	Is = BilinearInterpolate(img, sxs+x, sys+y)

print("--- %s seconds ---" % (time.time() - start))

start = time.time()
img = img.flatten()
for i in range(10):
	Is = interpolate.griddata((ys, xs), img, (sys+y, sxs+x))

print("--- %s seconds ---" % (time.time() - start))

