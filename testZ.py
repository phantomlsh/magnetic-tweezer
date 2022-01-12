import numpy as np
import time
import QI
import Z
import simulate
import matplotlib.pyplot as plt
from math import pi
from utils import BilinearInterpolate, Gaussian

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)

Rc = [] # real part
Φc = [] # phase angle
Ac = [] # amplitude

δc = np.arange(0, 10, 1)

for δ in δc:
  simulate.Generate(50, 50, 100+δ, 4, img)
  I = BilinearInterpolate(img, xs+x, ys+y)
  I = I.reshape((Nθ, Nr))
  It = Z.Tilde(QI.Profile(I))
  Rc.append(np.real(It))
  Φc.append(np.unwrap(np.angle(It)))
  Ac.append(np.abs(It))

δs = np.arange(0, 10, 0.01)
bias = []
for δ in δs:
  simulate.Generate(50, 50, 100+δ, 4, img)
  I = BilinearInterpolate(img, xs+x, ys+y)
  I = I.reshape((Nθ, Nr))
  It = Z.Tilde(QI.Profile(I))
  Ri = np.real(It)
  Φi = np.unwrap(np.angle(It))
  Ai = np.abs(It)
  χ2 = np.sum((Ri-Rc)**2, axis=1)
  ΔΦ = np.average(Φi-Φc, axis=1, weights=Ai*Ac)
  p = np.polynomial.polynomial.polyfit(δc, ΔΦ, 1)
  bias.append(-p[0]/p[1] - δ)

plt.plot(δs, bias, label="1 Calibration / unit")

plt.grid()
plt.legend()
plt.xlabel('Bias vs. test Z')
plt.ylabel('Bias')
plt.title('Test Z Position')
plt.show()
