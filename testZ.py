import numpy as np
import matplotlib.pyplot as plt
import time
import mm
import T

T.Init()
B = T.Bead(486, 203)
B.XY(mm.Get())

def imgSet(n):
    res = []
    for i in range(n):
        res.append(mm.Get())
        time.sleep(0.02)
    return res

# calibrate
sz = mm.GetZ()
for i in range(50):
    imgs = imgSet(10)
    z = mm.GetZ()
    B.Calibrate(imgs, z)
    mm.SetZ(z + 100)

mm.SetZ(sz)
# plt.title('Calibration')
# plt.imshow(np.flip(B.Rc, axis=0), cmap="gray")
# plt.show()

plt.title('Calibration Phi')
plt.imshow(np.flip(B.Φc, axis=0), cmap="gray")
plt.show()
# test

mm.SetZ(sz + 900)
for i in range(4):
    z = mm.GetZ()
    img = mm.Get()
    B.XY(img)
    print(B.Z(img) - z)
    mm.SetZ(z + 950)
    plt.axvline(x=z)

plt.grid()
plt.title('Delta Phi for 4 sampled z position')
plt.xlabel('Z(nm)')
plt.ylabel('Delta Phi')
plt.show()

mm.SetZ(sz)