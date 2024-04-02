"""
Notations:
n: number of beads
m: number of images in one packet
ζ: total number, = n * m
i: index of bead
j: index of image
μ: combined index, = i * m + j
tiFunc: Taichi kernels, called in Python scope
_func: Taichi functions, called in Taichi scope
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)
π = np.pi

"""
Global params initialization
@param r: expected bead radius in pixel
@param nθ: sampling number in polar direction
@param nr: sampling number in radial direction
"""


def SetParams(r=35, nr=80, nθ=80, maxn=30, maxm=2, maxz=100):
    global R, L, Nr, Nθ, Fr, Fθ
    R = r
    L = r * 2
    Nr = nr
    Nθ = nθ
    Fr = R / Nr
    Fθ = 2 * π / Nθ
    maxζ = maxn * maxm
    global _im, _p, _I, _J, _Z, _Φ, _A, _R, _cp, _Iq, _x, _cx
    _im = ti.field(dtype=ti.i32, shape=(maxζ, L + 10, L + 10))  # img grid
    _p = ti.field(dtype=ti.f32, shape=(maxζ, 3))  # points
    _I = ti.field(dtype=ti.f32, shape=(maxζ, Nr))  # intensity
    _J = ti.field(dtype=ti.f32, shape=(maxζ, Nr))  # imaginary part
    _Z = ti.field(dtype=ti.f32, shape=(maxz))  # calibration z
    _Φ = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr))  # calibration angle
    _A = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr))  # calibration amplitude
    _R = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr))  # calibration real
    _cp = ti.field(dtype=ti.i32, shape=(maxn, 3))  # calibration params
    # caches
    _Iq = ti.field(dtype=ti.f32, shape=(maxζ, Nr))  # fourier space
    _x = ti.field(dtype=ti.f32, shape=(maxζ, L))
    _cx = ti.field(dtype=ti.f32, shape=(maxζ, 50))


SetParams()


@ti.func  # Bilinear Interpolate
def _BI(μ: ti.i32, x: ti.f32, y: ti.f32) -> ti.f32:
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    xu = x1 - x
    xl = x - x0
    yu = y1 - y
    yl = y - y0
    return (
        xu * yu * _im[μ, y0, x0]
        + xu * yl * _im[μ, y1, x0]
        + xl * yu * _im[μ, y0, x1]
        + xl * yl * _im[μ, y1, x1]
    )


@ti.func  # fit a parabola
def _fitCenter(y0: ti.f32, y1: ti.f32, y2: ti.f32, y3: ti.f32, y4: ti.f32) -> ti.f32:
    a = (2 * y0 - y1 - 2 * y2 - y3 + 2 * y4) / 14
    b = -0.2 * y0 - 0.1 * y1 + 0.1 * y3 + 0.2 * y4
    return -b / a / 2


@ti.func  # fit a line
def _fitZero(
    x1: ti.f32,
    x2: ti.f32,
    x3: ti.f32,
    x4: ti.f32,
    x5: ti.f32,
    y1: ti.f32,
    y2: ti.f32,
    y3: ti.f32,
    y4: ti.f32,
    y5: ti.f32,
) -> ti.f32:
    a = 5
    c = x1 + x2 + x3 + x4 + x5
    d = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5
    D = a * d - c * c
    e = y1 + y2 + y3 + y4 + y5
    f = x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4 + x5 * y5
    b = (e * d - c * f) / D
    k = (f * a - e * c) / D
    return -b / k


@ti.func
def _cX(ζ: ti.i32, d: ti.i32):  # d = 0(X)|1(Y)
    for μ, k in ti.ndrange(ζ, 50):
        _cx[μ, k] = 0
    # normalize
    for μ, l in ti.ndrange(ζ, L):  # mean
        _cx[μ, 0] += _x[μ, l] / L
    for μ, l in ti.ndrange(ζ, L):  # std
        _cx[μ, 1] += (_x[μ, l] - _cx[μ, 0]) ** 2 / L
    for μ in ti.ndrange(ζ):
        _cx[μ, 1] = ti.sqrt(_cx[μ, 1])
    for μ, l in ti.ndrange(ζ, L):
        _x[μ, l] = (_x[μ, l] - _cx[μ, 0]) / _cx[μ, 1]
    # correlate
    for μ in ti.ndrange(ζ):
        _cx[μ, 0] = 0
        _cx[μ, 1] = 0
    for μ, k, l in ti.ndrange(ζ, 50, L):
        m = k - l + L - 26
        if m > 0 and m < L:
            _cx[μ, k] += _x[μ, l] * _x[μ, m]
    for μ in range(ζ):  # find max in correlate and fit
        x = 25
        for t in range(50):
            if _cx[μ, t] > _cx[μ, x]:
                x = t
        _p[μ, d] += (
            x
            - 25
            + _fitCenter(
                _cx[μ, x - 2], _cx[μ, x - 1], _cx[μ, x], _cx[μ, x + 1], _cx[μ, x + 2]
            )
        ) / 2


@ti.func
def _XY(ζ: ti.i32):
    for μ, t in ti.ndrange(ζ, (5, L + 5)):  # sample slice x
        _x[μ, t - 5] = (
            _BI(μ, t + _p[μ, 0], R + 4.3)
            + _BI(μ, t + _p[μ, 0], R + 5)
            + _BI(μ, t + _p[μ, 0], R + 5.7)
        )
    _cX(ζ, 0)
    for μ, t in ti.ndrange(ζ, (5, L + 5)):  # sample slice y
        _x[μ, t - 5] = (
            _BI(μ, R + 4.3, t + _p[μ, 1])
            + _BI(μ, R + 5, t + _p[μ, 1])
            + _BI(μ, R + 5.7, t + _p[μ, 1])
        )
    _cX(ζ, 1)


@ti.func
def _profile(ζ: ti.i32):
    for μ, r in ti.ndrange(ζ, Nr):
        _I[μ, r] = 0
    for μ, r, θ in ti.ndrange(ζ, Nr, Nθ):  # intensity profile
        x = R + 5 + _p[μ, 0] + Fr * r * ti.cos(θ * Fθ)
        y = R + 5 + _p[μ, 1] + Fr * r * ti.sin(θ * Fθ)
        _I[μ, r] += _BI(μ, x, y) / Nθ
    # normalize
    for μ in ti.ndrange(ζ):
        _cx[μ, 0] = 0  # mean
        _cx[μ, 1] = 0  # std
    for μ, r in ti.ndrange(ζ, Nr):
        _cx[μ, 0] += _I[μ, r] / Nr
    for μ, r in ti.ndrange(ζ, Nr):
        _cx[μ, 1] += (_I[μ, r] - _cx[μ, 0]) ** 2 / Nr
    for μ in ti.ndrange(ζ):
        _cx[μ, 1] = ti.sqrt(_cx[μ, 1])
    for μ, r in ti.ndrange(ζ, Nr):
        _I[μ, r] = (_I[μ, r] - _cx[μ, 0]) / _cx[μ, 1]


@ti.func
def _tilde(n: ti.i32, m: ti.i32):
    l = 2 * Nr
    ζ = n * m
    for μ, k in ti.ndrange(ζ, Nr):
        _Iq[μ, k] = 0
    for i, j, r in ti.ndrange(n, m, Nr):
        wl = _cp[i, 1]
        wr = _cp[i, 2]
        μ = i * m + j
        for k in range(wl, wr):  # Fourier transform with window
            _Iq[μ, k] += (
                _I[μ, r]
                * (
                    ti.cos(2 * π * k * (Nr - r - 1) / l)
                    + ti.cos(2 * π * k * (Nr + r) / l)
                )
                * (0.5 - 0.5 * ti.cos(2 * π * (k - wl) / (wr - wl - 1)))
            )
    for μ, r in ti.ndrange(ζ, Nr):
        _I[μ, r] = 0
        _J[μ, r] = 0
    for i, j in ti.ndrange(n, m):
        μ = i * m + j
        for k, r in ti.ndrange((_cp[i, 1], _cp[i, 2]), (_cp[i, 0], Nr)):
            _I[μ, r] += _Iq[μ, k] * ti.cos(2 * π * k * (r + Nr) / l) / l
            _J[μ, r] += _Iq[μ, k] * ti.sin(2 * π * k * (r + Nr) / l) / l


@ti.func
def _ΔΦ(μ: ti.i32, i: ti.i32, z: ti.i32) -> ti.f32:
    s = 0.0
    t = 0.0
    for r in range(_cp[i, 0], Nr):
        Δ = (ti.atan2(_J[μ, r], _I[μ, r]) - _Φ[i, z, r]) % (2 * π)
        if Δ > π:
            Δ -= 2 * π
        w = ti.sqrt(_I[μ, r] ** 2 + _J[μ, r] ** 2) * _A[i, z, r]
        t += w
        s += w * Δ
    return s / t


@ti.func
def __Z(n: ti.i32, m: ti.i32, nz: ti.i32):
    for i, j in ti.ndrange(n, m):
        μ = i * m + j
        rf = _cp[i, 0]
        z0 = 0
        minχ2 = 999999999.0
        for z in range(nz):
            χ2 = 0.0
            for r in range(rf, Nr):
                χ2 += (_I[μ, r] - _R[i, z, r]) ** 2
            if χ2 < minχ2:
                minχ2 = χ2
                z0 = z
        _p[μ, 2] = _fitZero(
            _Z[z0 - 2],
            _Z[z0 - 1],
            _Z[z0],
            _Z[z0 + 1],
            _Z[z0 + 2],
            _ΔΦ(μ, i, z0 - 2),
            _ΔΦ(μ, i, z0 - 1),
            _ΔΦ(μ, i, z0),
            _ΔΦ(μ, i, z0 + 1),
            _ΔΦ(μ, i, z0 + 2),
        )


@ti.kernel  # XY & profile
def tiProfile(ζ: ti.i32):
    _XY(ζ)  # iterate twice
    _XY(ζ)
    _profile(ζ)


@ti.kernel  # for ComputeCalibration
def tiTilde(n: ti.i32, m: ti.i32):
    _tilde(n, m)


@ti.kernel
def tiXYZ(n: ti.i32, m: ti.i32, nz: ti.i32):
    _XY(n * m)
    _XY(n * m)
    _profile(n * m)
    _tilde(n, m)
    __Z(n, m, nz)


# load data into taichi scope
# @return number of beads, number of images
def load(beads, imgs):
    n = len(beads)
    m = len(imgs)
    maxζ = _im.shape[0]
    if n * m > maxζ:
        raise Exception("Insufficient Capacity")
    im = []
    p = []
    for b in beads:  # n
        for img in imgs:  # m
            x = int(b.x)
            y = int(b.y)
            im.append(img[y - R - 5 : y + R + 5, x - R - 5 : x + R + 5])
            p.append([b.x - x, b.y - y, 0])
    im = np.pad(np.array(im, dtype=np.int32), ((0, maxζ - n * m), (0, 0), (0, 0)))
    p = np.pad(np.array(p, dtype=np.float32), ((0, maxζ - n * m), (0, 0)))
    _im.from_numpy(im)
    _p.from_numpy(p)
    return n, m


"""
Calculate XY Position
@param beads: list of beads
@param imgs: list of 2d array of image data
@return: [Bead0 Trace, Bead1 Trace, ...]
"""


def XY(beads, imgs):
    if len(beads) == 0:
        return
    n, m = load(beads, imgs)
    tiProfile(n * m)  # compute x, y, I in taichi scope
    p = _p.to_numpy()
    I = _I.to_numpy()
    bp = []
    for b in beads:
        bp.append([int(b.x), int(b.y)])
    res = []
    for i in range(n):
        res.append([])
        for j in range(m):
            b = beads[i]
            μ = i * m + j
            b.x = bp[i][0] + p[μ][0]
            b.y = bp[i][1] + p[μ][1]
            b.profile = I[μ]
            res[i].append([b.x, b.y])
    return res


"""
Calculate I and store
@param beads: list of beads
@param imgs: array of 2d array of image data
@param z: z position
"""


def Calibrate(beads, imgs, z):
    if len(beads) == 0:
        return
    for b in beads:
        b.l = []
    for img in imgs:
        XY(beads, [img])
        for b in beads:
            b.l.append(b.profile)
    for b in beads:
        b.Ic.append(np.average(b.l, axis=0))
        b.Zc.append(z)


"""
Finalize calibration by computing phase etc.
@param beads: list of beads
"""


def ComputeCalibration(beads):
    global Nz
    n = len(beads)
    if n == 0:
        return
    Nz = len(beads[0].Zc)
    maxn = _A.shape[0]
    maxz = _Z.shape[0]
    cp = []  # load calibration params
    for b in beads:
        b.Rc = []  # real part
        b.Φc = []  # phase angle
        b.Ac = []  # amplitude
        cp.append([b.rf, b.w[0], b.w[1]])
    cp = np.pad(cp, ((0, maxn - n), (0, 0)))
    _cp.from_numpy(cp)
    for z in range(Nz):
        I = []
        for i in range(_I.shape[0]):
            if i < n:
                I.append(beads[i].Ic[z])
            else:
                I.append(np.zeros(Nr))
        _I.from_numpy(np.array(I, dtype=np.float32))
        tiTilde(n, 1)
        I = _I.to_numpy()
        J = _J.to_numpy()
        for i in range(n):
            b = beads[i]
            b.Rc.append(I[i])
            b.Φc.append(np.arctan2(J[i], I[i]))
            b.Ac.append(np.sqrt(I[i] ** 2 + J[i] ** 2))
    Zc = []
    Φc = []
    Ac = []
    Rc = []
    for b in beads:
        b.Rc = np.array(b.Rc)
        b.Ac = np.array(b.Ac)
        b.Φc = np.array(b.Φc)
        Zc = b.Zc  # 1d array
        Φc.append(b.Φc)
        Ac.append(b.Ac)
        Rc.append(b.Rc)
    Zc = np.pad(Zc, (0, maxz - Nz))
    Φc = np.pad(Φc, ((0, maxn - n), (0, maxz - Nz), (0, 0)))
    Ac = np.pad(Ac, ((0, maxn - n), (0, maxz - Nz), (0, 0)))
    Rc = np.pad(Rc, ((0, maxn - n), (0, maxz - Nz), (0, 0)))
    _Z.from_numpy(Zc.astype(np.float32))
    _Φ.from_numpy(Φc.astype(np.float32))
    _A.from_numpy(Ac.astype(np.float32))
    _R.from_numpy(Rc.astype(np.float32))


"""
Calculate XYZ Position (cover XY)
@param beads: list of beads
@param imgs: list of 2d array of image data
@return: [Bead0 Trace, Bead1 Trace, ...]
"""


def XYZ(beads, imgs):
    if len(beads) == 0:
        return
    n, m = load(beads, imgs)
    tiXYZ(n, m, Nz)
    p = _p.to_numpy()
    # p[np.abs(p) > 10] = 0 # filter outlier
    bp = []
    for b in beads:
        bp.append([int(b.x), int(b.y)])
    res = []
    for i in range(n):
        res.append([])
        for j in range(m):
            b = beads[i]
            μ = i * m + j
            b.x = bp[i][0] + p[μ][0]
            b.y = bp[i][1] + p[μ][1]
            b.z = p[μ][2]
            res[i].append([b.x, b.y, b.z])
    return res


"""
Interface for beads
@param rf: forget radius
@param w: window in Fourier space
"""


class Bead:
    def __init__(self, x, y, rf=10, w=[2, 40]):
        self.x = x
        self.y = y
        self.z = 0
        self.rf = rf
        self.w = w
        # self calibration
        self.Ic = []  # Intensity Profiles
        self.Zc = []  # Z values

    def __repr__(self):
        return f"Bead({self.x}, {self.y}, {self.z}, rf={self.rf}, w=[{self.w[0]}, {self.w[1]}])"

    def __str__(self):
        return f"Bead({self.x}, {self.y}, {self.z}, rf={self.rf}, w=[{self.w[0]}, {self.w[1]}])"
