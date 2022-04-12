import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3)

ds = np.arange(-2, 2, 0.01)
bias = []

for d in ds:
    y = (x - d)**2
    p = np.polynomial.polynomial.polyfit(x, y, 2)
    bias.append(-p[1]/(2*p[2]) - d)

plt.plot(ds, bias)
plt.show()
