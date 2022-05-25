import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

maxN = 100
π = np.pi
im = ti.field(dtype=ti.i32, shape=(1000, 1000))
ps = ti.Vector.field(dtype=ti.f32, n=2, shape=(maxN))
Is = ti.field(dtype=ti.f32, shape=(maxN, 100))

def SetParams(H, W, r=40, nr=80, nθ=80):
    global R, Nr, Nθ, Fr, Fθ, im, ps, Is
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ
    im = ti.field(dtype=ti.i32, shape=(H, W))
    ps = ti.Vector.field(dtype=ti.f32, n=2, shape=(maxN))
    Is = ti.field(dtype=ti.f32, shape=(maxN, Nr))

@ti.kernel
def tiBI(n: ti.i32):
    for i, r in ti.ndrange(n, Nr):
        Is[i, r] = 0
        for θ in range(Nθ):
            x = ps[i][0] + Fr * r * ti.cos(θ * Fθ)
            y = ps[i][0] + Fr * r * ti.sin(θ * Fθ)

            x0 = int(x)
            y0 = int(y)
            x1 = x0 + 1
            y1 = y0 + 1

            xu = x1 - x
            xl = x - x0
            yu = y1 - y
            yl = y - y0

            wa = xu * yu
            wb = xu * yl
            wc = xl * yu
            wd = xl * yl

            Is[i, r] += (wa*im[y0, x0] + wb*im[y1, x0] + wc*im[y0, x1] + wd*im[y1, x1]) / Nθ

def Profile(beads, img):
    im.from_numpy(img.astype(int))
    n = len(beads)
    for i in range(n):
        ps[i][0] = beads[i].x
        ps[i][1] = beads[i].y
    tiBI(n)
    res = Is.to_numpy()
    for i in range(n):
        beads[i].profile = res[i]
