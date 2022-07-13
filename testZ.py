import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T

beads = []
zstep = 50

circles = utils.HoughCircles(mm.Get(), 15, 30)
# for c in circles:
#     if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
#         beads.append(T.Bead(c[0], c[1]))

beads = [T.Bead(circles[0][0], circles[0][1]), T.Bead(circles[1][0], circles[1][1])]

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
    mm.SetZ(z + zstep)

T.ComputeCalibration(beads)

plt.title('Calibration R')
plt.imshow(np.flip(beads[0].Rc, axis=0), cmap="gray")
plt.show()

# test

zs = []
z0s = []

for z in range(80):
    mm.SetZ(sz + (z+10)*zstep)
    for i in range(100):
        img = mm.Get()
        T.XYZ(beads, img)
        z0s.append(beads[0].z)
        zs.append(beads[0].z - beads[1].z)

xs = []
ys = []
yerr = []
for z in range(80):
    x = sz + (z+10)*zstep
    data = z0s[z*100:(z*100+100)]
    xs.append(x)
    ys.append(np.mean(data) - x)
    yerr.append(np.std(data))

plt.plot(z0s, marker="o")
plt.grid()
plt.show()

plt.grid()
plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
plt.title('Bias in Z tracking')
plt.xlabel('Z(nm)')
plt.ylabel('Bias(nm)')
plt.show()

mm.SetZ(sz)