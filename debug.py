import numpy as np
import time
import simulate
import matplotlib.pyplot as plt
import T as T
import N as N

X = 100
Y = 100
shape = (700, 756)
imgc = []
for i in range(50):
    img = np.ndarray(shape)
    simulate.Generate(X, Y, 20 + i, 4, img)
    imgc.append(img)
img = np.ndarray(shape)
simulate.Generate(X, Y, 50, 4, img)

plt.imshow(img)
plt.show()

beads1 = [N.Bead(X, Y)]

for i in range(50):
    N.Calibrate(beads1, [imgc[i]], i)

N.ComputeCalibration(beads1)

plt.subplot(2, 3, 1)
plt.title("Profile")
plt.imshow(np.flip(beads1[0].Ic, axis=0), cmap="gray")
plt.subplot(2, 3, 2)
plt.title("Real Part")
plt.imshow(np.flip(beads1[0].Rc, axis=0), cmap="gray")
plt.subplot(2, 3, 3)
plt.title("Phase")
plt.imshow(np.flip(beads1[0].Φc, axis=0))

beads2 = [T.Bead(X, Y)]

for i in range(50):
    T.Calibrate(beads2, [imgc[i]], i)

T.ComputeCalibration(beads2)

plt.subplot(2, 3, 4)
plt.title("Profile")
plt.imshow(np.flip(beads2[0].Ic, axis=0), cmap="gray")
plt.subplot(2, 3, 5)
plt.title("Real Part")
plt.imshow(np.flip(beads2[0].Rc, axis=0), cmap="gray")
plt.subplot(2, 3, 6)
plt.title("Phase")
plt.imshow(np.flip(beads2[0].Φc, axis=0))
plt.show()
