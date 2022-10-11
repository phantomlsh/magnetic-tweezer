import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import N as T

beads = []
zcs = np.arange(0, 10000, 100)

circles = utils.HoughCircles(mm.Get(), 15, 30)
# for c in circles:
#     if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
#         beads.append(T.Bead(c[0], c[1]))

# beads = [T.Bead(circles[0][0], circles[0][1]), T.Bead(circles[1][0], circles[1][1])]

beads = [T.Bead(179, 378), T.Bead(307, 231)]

n = len(beads)
print(n, "Beads:", beads)

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

plt.title('Calibration R')
plt.imshow(np.flip(beads[0].Rc, axis=0), cmap="gray")
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

# sample
mm.SetZ(sz + 5000)
f = open("Z.dat", "w")
time.sleep(0.2)
last = time.time()
for t in range(10000):
    img = mm.Get()
    T.XYZ(beads, [img])
    f.write(str(beads[0].z) + ' ' + str(beads[1].z) + '\n')
    last = time.time()
    time.sleep(0.02)

mm.SetZ(sz)
f.close()