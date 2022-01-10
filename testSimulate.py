import numpy as np
import time
import QI
import simulate
from utils import BilinearInterpolate, NormalizeArray

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)

simulate.Generate(50+0.1, 50, 50, 4, img)
Δs = []

start = time.time()
for i in range(500):
	dx = 0
	dy = 0
	s = []
	for i in range(3):
		Is = BilinearInterpolate(img, xs+x+dx, ys+y+dy)
		Is = Is.reshape((Nθ, Nr))
		Δx, Δy = QI.XY(Is, Nθ)
		dx += Δx * 0.8
		dy += Δy * 0.8
	
	Δs.append(dx - 0.1)

print("--- %s seconds ---" % (time.time() - start))
print("mean =", np.mean(Δs), "standard deviation =", np.std(Δs))
