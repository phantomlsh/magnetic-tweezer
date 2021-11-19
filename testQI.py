import numpy as np
import time
import QI
import simulate
import position
import matplotlib.pyplot as plt
from utils import BilinearInterpolate

img = np.ndarray((100, 100))
x = 50
y = 50
r = 40
N = 80
Nθ = 80
Nr = 80
δs = np.arange(0, 5, 0.01)
xs, ys = QI.SamplePoints(N, Nθ, Nr)
Δs = []

start = time.time()
for δ in δs:
	simulate.Generate(50+δ, 50, 50, 4, img)
	dx = 0
	dy = 0
	for i in range(6):
		Is = BilinearInterpolate(img, xs+x+dx, ys+y+dy)
		Is = Is.reshape((Nθ, Nr))
		Δ, Δy = QI.XY(Is, dx, dy, Nθ)
		dx = Δ

	Δs.append(Δ - δ)

print("--- %s seconds ---" % (time.time() - start))
print(np.std(Δs))
