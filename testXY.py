import numpy as np
import time
import QI
import simulate
import matplotlib.pyplot as plt
import utils

img = np.ndarray((100, 100))

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)

δs = np.arange(0, 5, 0.1)
Δs = []
Δs2 = []
xs = []

def XYF(img, x, y, it):
	print('it =', it)
	xl = int(x)
	yl = int(y)
	r = 30
	dx = x-xl
	dy = y-yl
	for i in range(it):
		dx = dx + utils.SymmetryCenter(np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0), dx)
		dy = dy + utils.SymmetryCenter(np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1), dy)
		print(dx, dy)
	return xl+dx, yl+dy

start = time.time()
# for δ in δs:
# 	simulate.Generate(50+δ, 50, 50, 4, img)
# 	x, y = XYF(img, 50, 50, 1)
# 	Δs.append(x - 50 - δ)
# 	x2, y2 = XYF(img, 50, 50, 2)
# 	Δs2.append(x2 - 50 - δ)

# print("--- %s seconds ---" % (time.time() - start))
# print("mean =", np.mean(Δs), "standard deviation =", np.std(Δs))
# plt.plot(δs, Δs)
# plt.plot(δs, Δs2)
# plt.show()

simulate.Generate(50, 50, 50, 4, img)
XYF(img, 49, 50, 110)
