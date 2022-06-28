import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T

beads = []

circles = utils.HoughCircles(mm.Get(), 15, 30)
for c in circles:
    if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
        beads.append(T.Bead(c[0], c[1]))

n = len(beads)
print(n, "Beads:", beads)

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
plt.imshow(np.flip(beads[0].Φc, axis=0), cmap="gray")
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
plt.scatter(range(len(zs)), zs)
plt.title('Delta Z between two beads')
plt.xlabel('Frame')
plt.ylabel('Delta Z(nm)')
plt.show()

print(np.std(zs))

mm.SetZ(sz)