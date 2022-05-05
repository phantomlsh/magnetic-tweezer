import numpy as np
import taichi as ti
import time

ti.init(arch=ti.gpu, default_fp=ti.f32)

@ti.kernel
def sum1000(x: ti.types.ndarray()) -> ti.f32:
    s = 0.0
    for i in range(1000):
        s += x[i]
    return s

def test(x):
    sum1000(x)
    start = time.time()
    for i in range(1000):
        sum1000(x)
    print("duration(s): ", time.time() - start)
    
test(np.arange(0, 10, 0.001, dtype=np.float32))
test(np.zeros(1000, dtype=np.float32))
test(np.zeros(1000, dtype=int))