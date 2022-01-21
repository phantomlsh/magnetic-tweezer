import numpy as np
import time
import QI
import simulate
import matplotlib.pyplot as plt
from utils import BilinearInterpolate

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)

δs = np.arange(0, 5, 0.01)
Δs = []

start = time.time()
for δ in δs:
	simulate.Generate(50+δ, 50, 50, 4, img)
	dx = 0
	dy = 0
	for i in range(3):
		I = BilinearInterpolate(img, xs+x+dx, ys+y+dy)
		I = I.reshape((Nθ, Nr))
		Δx, Δy = QI.XY(I, Nθ)
		dx += Δx * 0.8
		dy += Δy * 0.8
	
	Δs.append(dx - δ)

print("--- %s seconds ---" % (time.time() - start))
print("mean =", np.mean(Δs), "standard deviation =", np.std(Δs))
