import numpy as np
import matplotlib.pyplot as plt
import time
import mm
import T

beads = [T.Bead(507, 381), T.Bead(477, 454)]
for i in range(3):
    beads.append(T.Bead(507, 381))
T.XY(beads, mm.Get())

def imgSet(n):
    res = []
    for i in range(n):
        res.append(mm.Get())
        time.sleep(0.02)
    return res

# calibrate
sz = mm.GetZ()
for i in range(50):
    imgs = imgSet(5)
    z = mm.GetZ()
    T.Calibrate(beads, imgs, z)
    mm.SetZ(z + 100)

T.ComputeCalibration(beads)

plt.title('Calibration R')
plt.imshow(np.flip(beads[0].Rc, axis=0), cmap="gray")
plt.show()

# test

mm.SetZ(sz + 900)
# for i in range(4):
#     z = mm.GetZ()
#     img = mm.Get()
#     T.XY(beads, img)
#     T.Z(beads, img)
#     print(beads[0].z - z)
#     mm.SetZ(z + 950)
#     plt.axvline(x=z)

# # plt.grid()
# # plt.title('Delta Phi for 4 sampled z position')
# # plt.xlabel('Z(nm)')
# # plt.ylabel('Delta Phi')
# # plt.show()

start = time.time()
zs = []
for i in range(1000):
    img = mm.Get()
    T.XY(beads, img)
    T.Z(beads, img)
    zs.append(beads[0].z - beads[1].z)

print('time =', time.time() - start)

plt.grid()
plt.plot(zs, marker="o")
plt.title('Delta Z between two beads')
plt.xlabel('Frame')
plt.ylabel('Delta Z(nm)')
plt.show()

print(np.std(zs))

mm.SetZ(sz)