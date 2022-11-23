import numpy as np
import matplotlib.pyplot as plt
import time, mm
import cv2 as cv
import N as T

zcs = np.arange(0, 10000, 100)
beads = [T.Bead(360, 273)]
n = len(beads)

sz = mm.GetZ()
T.XY(beads, [mm.Get()])

# calibrate
for i in range(len(zcs)):
    z = zcs[i]
    imgc = []
    mm.SetZ(z + sz)
    time.sleep(0.2)
    for t in range(10):
        imgc.append(mm.Get())
        time.sleep(0.02)
    T.Calibrate(beads, imgc, z + sz)

mm.SetZ(sz)
T.ComputeCalibration(beads)
# cv.imwrite("data/C1.png", np.array(beads[0].Ic))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Real Part")
ax1.imshow(np.flip(beads[0].Rc, axis=0), cmap="gray")
ax2.set_title("Phase")
ax2.imshow(np.flip(beads[0].Φc, axis=0))
plt.show()

# test
zts = np.arange(500, 9500, 50)
z0s = []
for z in zts:
    mm.SetZ(sz + z)
    time.sleep(0.2)
    for t in range(100):
        img = mm.Get()
        T.XYZ(beads, [img])
        z0s.append(beads[0].z)

xs = []
ys = []
yerr = []
for i, z in enumerate(zts):
    x = sz + z
    data = z0s[i*100:(i*100+100)]
    xs.append(x)
    ys.append(np.mean(data) - x)
    yerr.append(np.std(data))

plt.grid()
plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
plt.title('Bias in Z tracking')
plt.xlabel('Z(nm)')
plt.ylabel('Bias(nm)')
plt.show()

plt.title("Phase Difference in Neighborhood")
print(T.ΔΦs)
plt.imshow(T.ΔΦs)
plt.show()

mm.SetZ(sz)
