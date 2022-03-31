import numpy as np
import matplotlib.pyplot as plt
import time
import mm
import T

T.Init()
B = T.Bead(342, 305)
B.XY(mm.Get())

# calibrate
sz = mm.GetZ()
for i in range(50):
    img = mm.Get()
    z = mm.GetZ()
    B.XY(img)
    B.Calibrate(img, z)
    mm.SetZ(z + 100)

mm.SetZ(sz)
plt.title('Calibration')
plt.imshow(np.flip(B.Rc, axis=0), cmap="gray")
plt.show()
# test

mm.SetZ(sz + 900)
for i in range(4):
    z = mm.GetZ()
    plt.axvline(x=z)
    img = mm.Get()
    B.XY(img)
    print(B.Z(img) - z)
    mm.SetZ(z + 950)

plt.grid()
plt.title('Delta Phi for 4 sampled z position')
plt.xlabel('Z(nm)')
plt.ylabel('Delta Phi')
plt.show()

mm.SetZ(sz)