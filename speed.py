import numpy as np
import taichi as ti
import time

ti.init(arch=ti.gpu)

@ti.kernel
def tiChiSquare(Rc: ti.types.ndarray(), R: ti.types.ndarray(), res: ti.types.ndarray()):
    n = Rc.shape[0]
    l = Rc.shape[1]
    mini = -1
    mins = 2.1e10
    for i in range(n):
        res[i] = 0
        for j in range(l):
            res[i] += (Rc[i, j] - R[j]) ** 2

def minχ2(Rc, R):
    χ2 = np.zeros(len(ys))
    tiChiSquare(ys, y, χ2)
    return np.argmin(χ2)

xs = np.arange(0, 10, 0.01)
ys = []
for i in range(100):
    ys.append(np.sin(xs + i/100))
ys = np.array(ys)
y = np.sin(xs + 20/100)

χ2 = np.sum((y-ys)**2, axis=1)

print(np.argmin(χ2))
print(minχ2(ys, y))

start = time.time()

for i in range(1000):
    minχ2(ys, y)

print(time.time() - start)

start = time.time()

for i in range(1000):
    χ2 = np.sum((y-ys)**2, axis=1)
    np.argmin(χ2)

print(time.time() - start)
