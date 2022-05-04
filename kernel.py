import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

π = np.pi

def SetParams(r=40, nr=80, nθ=80):
    global R, Nr, Nθ, Fr, Fθ
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ

@ti.kernel
def tiBI(im: ti.types.ndarray(), ps: ti.types.ndarray(), res: ti.types.ndarray()):
    n = ps.shape[0]
    ll = n * Nr # total result length
    for l in range(ll):
        r = l % Nr # index of ring
        i = l // Nr # index of beads
        for θ in range(Nθ):
            x = ps[i, 0] + Fr * r * ti.cos(θ * Fθ)
            y = ps[i, 1] + Fr * r * ti.sin(θ * Fθ)

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

def Profile(beads, img):
    n = len(beads)
    ps = []
    for b in beads:
        ps.append([b.x, b.y])
    Is = np.zeros(Nr * n)
    tiBI(img.astype(int), np.array(ps), Is)
    Is = Is.reshape((n, Nr))
    for i in range(n):
        beads[i].profile = Is[i]
    return Is
