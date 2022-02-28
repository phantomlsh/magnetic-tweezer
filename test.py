import numpy as np
import matplotlib.pyplot as plt
from utils import BilinearInterpolate, Gaussian, SymmetryCenter
from scipy import signal

xs = np.arange(-10, 10.1, 0.1)
delta = np.arange(-1, 1, 0.01)
bias = []
for d in delta:
	I = Gaussian(xs, d, 4) * np.cos((xs-d))
	bias.append(SymmetryCenter(I)/10 - d)

plt.plot(delta, bias)
plt.grid()
plt.show()
# plt.plot(xs, signal.correlate(np.flip(I), I, mode="same"))
# plt.xticks(np.arange(-10, 10.1, 1))
# plt.grid()
# plt.show()