import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import N as T

beads = []

circles = utils.HoughCircles(mm.Get(), 15, 30)
# for c in circles:
#     if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
#         beads.append(T.Bead(c[0], c[1]))

beads = [T.Bead(circles[0][0], circles[0][1])]

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
for i in range(100):
    imgs = imgSet(5)
    z = mm.GetZ()
    T.Calibrate(beads, imgs, z)
    mm.SetZ(z + 100)

T.ComputeCalibration(beads)

plt.title('Calibration R')
plt.imshow(np.flip(beads[0].Rc, axis=0), cmap="gray")
plt.show()

# test

zs = []

for z in range(80):
    mm.SetZ(sz + 1000 + z*100)
    for i in range(100):
        img = mm.Get()
        T.XYZ(beads, img)
        zs.append(beads[0].z)

xs = []
ys = []
yerr = []
for z in range(80):
    x = sz + 1000 + z*100
    data = zs[z*100:(z*100+100)]
    xs.append(x)
    ys.append(np.mean(data) - x)
    yerr.append(np.std(data))

plt.grid()
plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
plt.title('Bias in Z tracking')
plt.xlabel('Z(nm)')
plt.ylabel('Bias(nm)')
plt.show()

mm.SetZ(sz)