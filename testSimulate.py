import numpy as np
from scipy import interpolate
import time
import QI
import simulate
import matplotlib.pyplot as plt
import utils
from numba import jit

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)
simulate.Generate(50+0.5, 50, 50, 4, img)

@jit(nopython=True, fastmath=True, cache=True)
def eval(f):
	for i in range(1000):
		Is = np.array([float(f(*p)) for p in zip(x+xs, y+ys)])
	return Is

start = time.time()
f = interpolate.interp2d(np.arange(0, 100), np.arange(0, 100), img)
Is = eval(f)
print(time.time() - start)

plt.imshow(Is.reshape((80, 80)), cmap="gray")
plt.show()
