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
def tiBI(ims: ti.types.ndarray(), ps: ti.types.ndarray(), res: ti.types.ndarray()):
    n = ps.shape[0]
    ll = n * Nr # total result length
    for l in range(ll):
        r = l % Nr # index of ring
        i = l // Nr # index of beads
        for θ in range(Nθ):
            x = ps[i, 0] + Fr * r * ti.cos(θ * Fθ) + R + 1
            y = ps[i, 1] + Fr * r * ti.sin(θ * Fθ) + R + 1

            x0 = int(x)
            y0 = int(y)
            x1 = x0 + 1
            y1 = y0 + 1

            Ia = ims[i, y0, x0]
            Ib = ims[i, y1, x0]
            Ic = ims[i, y0, x1]
            Id = ims[i, y1, x1]

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
    ims = []
    for b in beads:
        x0 = int(b.x)
        y0 = int(b.y)
        ims.append(img[y0-R-1:y0+R+1, x0-R-1:x0+R+1])
        ps.append([b.x-x0, b.y-y0])
    Is = np.zeros(Nr * n, dtype=np.float32)
    tiBI(np.array(ims, dtype=np.float32), np.array(ps, dtype=np.float32), Is)
    Is = Is.reshape((n, Nr))
    for i in range(n):
        beads[i].profile = Is[i]
    return Is
