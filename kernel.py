import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
ti.init(arch=ti.gpu)

maxN = 100
π = np.pi

im = ti.field(dtype=ti.i32, shape=(1000, 1000))
ps = ti.Vector.field(dtype=ti.f32, n=2, shape=(maxN))
Is = ti.field(dtype=ti.f32, shape=(maxN, 100))

def SetParams(r=40, nr=80, nθ=80):
    global R, Nr, Nθ, Fr, Fθ, ps, Is
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ
    ps = ti.Vector.field(dtype=ti.f32, n=2, shape=(maxN))
    Is = ti.field(dtype=ti.f32, shape=(maxN, Nr))

@ti.kernel
def tiBI(n: ti.i32):
    for i, r in ti.ndrange(n, Nr):
        Is[i, r] = 0
        for θ in range(Nθ):
            x = ps[i][0] + Fr * r * ti.cos(θ * Fθ)
            y = ps[i][1] + Fr * r * ti.sin(θ * Fθ)
            x0 = int(x)
            y0 = int(y)
            x1 = x0 + 1
            y1 = y0 + 1
            xu = x1 - x
            xl = x - x0
            yu = y1 - y
            yl = y - y0
            Is[i, r] += (xu*yu*im[y0, x0] + xu*yl*im[y1, x0] + xl*yu*im[y0, x1] + xl*yl*im[y1, x1]) / Nθ

def Profile(beads, img):
    global im
    if (im.shape[0] != img.shape[0] or im.shape[1] != img.shape[1]):
        im = ti.field(dtype=ti.i32, shape=(img.shape[0], img.shape[1]))
    im.from_numpy(img.astype(int))
    n = len(beads)
    for i in range(n):
        ps[i][0] = beads[i].x
        ps[i][1] = beads[i].y
    tiBI(n)
    res = Is.to_numpy()
    for i in range(n):
        beads[i].profile = res[i]
