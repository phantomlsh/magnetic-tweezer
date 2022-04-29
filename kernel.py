import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

π = np.pi

@ti.kernel
def tiFillSPs():
    fr = R/Nr
    fθ = 2*π/Nθ
    for r in range(Nr):
        for θ in range(Nθ):
            SPs[r, θ][0] = fr * r * ti.cos(θ * fθ)
            SPs[r, θ][1] = fr * r * ti.sin(θ * fθ)

def SetSamplePoint(r=40, nr=80, nθ=80):
    global R, Nr, Nθ, SPs
    R = r
    Nr = nr
    Nθ = nθ
    SPs = ti.Vector.field(n=2, dtype=ti.f32, shape=(nr,nθ))
    tiFillSPs()

SetSamplePoint()

@ti.kernel
def tiBI(im: ti.types.ndarray(), n: int, xs: ti.types.ndarray(), ys: ti.types.ndarray(), res: ti.types.ndarray()):
    ll = n * Nr # total result length
    for l in range(ll):
        r = l % Nr # index of ring
        i = l // Nr # index of beads
        for θ in range(Nθ):
            x = xs[i] + SPs[r, θ][0]
            y = ys[i] + SPs[r, θ][1]

            x0 = int(x)
            y0 = int(y)
            x1 = x0 + 1
            y1 = y0 + 1

            Ia = im[y0, x0]
            Ib = im[y1, x0]
            Ic = im[y0, x1]
            Id = im[y1, x1]

            xu = x1 - x
            xl = x - x0
            yu = y1 - y
            yl = y - y0

            wa = xu * yu
            wb = xu * yl
            wc = xl * yu
            wd = xl * yl

            res[l] += wa*Ia + wb*Ib + wc*Ic + wd*Id

        res[l] /= Nθ

def Profile(img, beads):
    n = len(beads)
    xs = []
    ys = []
    for b in beads:
        xs.append(b.x)
        ys.append(b.y)
    Is = np.zeros(Nr * n)
    tiBI(img, n, np.array(xs), np.array(ys), Is)
    return Is.reshape((n, Nr))
