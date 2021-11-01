import numpy as np
from math import floor
from scipy import signal
import matplotlib.pyplot as plt

delta = 0.899
x = np.arange(-20, 20.1, 0.1)
y = np.cos(x-delta)*np.exp(-0.1*np.abs(x-delta))
co = signal.correlate(y, np.flip(y), mode="same")
l = len(x)
c = floor(len(x)/2)
maxi = np.argmax(co[c-100:c+100])-100+c
p = np.polyfit(range(maxi-3, maxi+4, 1), co[maxi-3:maxi+4], 2)
print(-p[1]/(4*p[0])-c/2)
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(y, label="y")
axs[0].plot(np.flip(y), label="y_{flip}")
axs[0].grid()
axs[0].legend()
axs[1].plot(co, label="correlation")
axs[1].legend()
axs[1].grid()
#plt.show()
