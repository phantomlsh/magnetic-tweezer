import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, floor, pi as π
from utils import NormalizeArray
from numba import jit

import position

@jit(nopython=True)
def Generate(X, Y, Z, λ, Is):
	L = Is.shape[0]
	for x in range(L):
		for y in range(L):
			r = sqrt((x-X)**2 + (y-Y)**2 + Z**2)
			n = (r-Z)/λ
			ϕ = 2*π*(n - floor(n) - 0.5)
			Is[y][x] = cos(ϕ/2) * Z**2 / (r**4)

Is = np.ndarray((100, 100))

x = 50
y = 50
r = 40
δs = np.arange(0, 5, 0.01)
Δ1s = []
Δ2s = []
Δ3s = []

for δ in δs:
	Generate(50+δ, 50, 50, 4, Is)
	Δ1 = position.SymmetryCenter(np.mean(Is[y-2:y+3, x-r:x+r], axis=0), 0)
	d = Δ1
	Δ2 = position.SymmetryCenter(np.mean(Is[y-2:y+3, x-r:x+r], axis=0), d) + d
	d = Δ2
	Δ3 = position.SymmetryCenter(np.mean(Is[y-2:y+3, x-r:x+r], axis=0), d) + d
	Δ1s.append(Δ1-δ)
	Δ2s.append(Δ2-δ)
	Δ3s.append(Δ3-δ)

plt.plot(δs, Δ1s)
plt.plot(δs, Δ2s)
plt.plot(δs, Δ3s)
plt.show()