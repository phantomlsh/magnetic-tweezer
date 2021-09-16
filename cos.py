import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

delta = 1
x = np.arange(-20, 20.1, 0.1)
y = np.cos(x-delta)*np.exp(-0.1*np.abs(x-delta))
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(x, y, label="y")
axs[0].plot(x, np.flip(y), label="y_{flip}")
axs[0].grid()
axs[0].legend()
axs[1].plot(x, signal.correlate(y, np.flip(y), mode="same"), label="correlation")
axs[1].legend()
axs[1].grid()
plt.show()
