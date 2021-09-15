import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

delta = 1
x = np.arange(-20, 20.1, 0.1)
y = np.cos(x-delta)*np.exp(-0.1*np.abs(x-delta))
plt.plot(x, y, label="y")
plt.plot(x, np.flip(y), label="yflip")
plt.legend()
plt.show()
plt.plot(x, signal.correlate(y, np.flip(y), mode="same"))
plt.show()
