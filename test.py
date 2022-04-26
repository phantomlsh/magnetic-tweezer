import numpy as np
import matplotlib.pyplot as plt
import utils

x = np.arange(-5, 5)

ds = np.arange(-3, 3, 0.01)
bias = []

for d in ds:
    y = np.cos((x-d)/10)
    p = np.polynomial.polynomial.polyfit(x, y, 2)
    bias.append(-p[1]/(2*p[2]) - d)

plt.plot(ds, bias)
plt.grid()
plt.show()
