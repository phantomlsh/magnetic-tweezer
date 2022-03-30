import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils, Z
import T

T.Init()
B = T.Bead(350, 290)

# calibrate
sz = mm.GetZ()
for i in range(50):
    img = mm.Get()
    z = mm.GetZ()
    B.XY(img)
    B.Calibrate(img, z)
    mm.SetZ(z + 100)

mm.SetZ(sz)
plt.imshow(np.flip(B.Rc, axis=0))
plt.show()
# test

mm.SetZ(sz + 200)
zs = []
for i in range(4):
    z = mm.GetZ()
    zs.append(z)
    img = mm.Get()
    B.XY(img)
    print(B.Z(img) - z)
    mm.SetZ(z + 200)

plt.scatter(zs, np.zeros(4))
plt.grid()
plt.show()

mm.SetZ(sz)