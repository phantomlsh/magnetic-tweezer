import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, Z
import T

# parameters
T.Init()

T.Calibrate(mm.Get, mm.GetZ, mm.SetZ, 200, 320)

# test
sz = mm.GetZ()
mm.SetZ(sz + 300)
zs = []
bias = []
for i in range(4):
	z = mm.GetZ()
	zs.append(z)
	img = mm.Get()
	x, y = T.XY(img, 200, 320)
	It = Z.Tilde(T.Profile(img, x, y))
	bias.append(Z.Z(It) - z)
	mm.SetZ(z + 200)

print(bias)
plt.scatter(zs, np.zeros(4))
plt.grid()
plt.show()

mm.SetZ(sz)