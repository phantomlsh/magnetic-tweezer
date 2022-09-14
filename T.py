"""
Notations:
n: number of beads
m: number of images in one packet
ζ: total number, = n * m
i: index of bead
j: index of image
μ: combined index, = j * n + i
tiFunc: Taichi kernels, called in Python scope
_func: Taichi functions, called in Taichi scope
"""

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)
π = np.pi

"""
Global params initialization
@param r: expected bead radius in pixel
@param nθ: sampling number in polar direction
@param nr: sampling number in radial direction
"""
def SetParams(r=40, nr=80, nθ=80, maxn=30, maxm=2, maxz=100):
    global R, L, freq, Nr, Nθ, Fr, Fθ
    R = r
    L = r * 2
    freq = np.fft.rfftfreq(L*2)
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ
    maxζ = maxn * maxm
    global _im, _p, _I, _J, _Z, _Φ, _A, _R, _Iq, _x, _cx
    _im = ti.field(dtype=ti.i32, shape=(maxζ, 2*R+10, 2*R+10)) # img grid
    _p = ti.field(dtype=ti.f32, shape=(maxζ, 3)) # points
    _I = ti.field(dtype=ti.f32, shape=(maxζ, Nr)) # intensity
    _J = ti.field(dtype=ti.f32, shape=(maxζ, Nr)) # imaginary part
    _Z = ti.field(dtype=ti.f32, shape=(maxz)) # calibration z
    _Φ = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration angle
    _A = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration amplitude
    _R = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration real
    # caches
    _Iq = ti.field(dtype=ti.f32, shape=(maxζ, Nr)) # fourier space
    _x = ti.field(dtype=ti.f32, shape=(maxζ, 2*R))
    _cx = ti.field(dtype=ti.f32, shape=(maxζ, 30))

SetParams()

@ti.func # Bilinear Interpolate
def _BI(μ: ti.i32, x: ti.f32, y: ti.f32) -> ti.f32:
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    xu = x1 - x
    xl = x - x0
    yu = y1 - y
    yl = y - y0
    return xu*yu*_im[μ, y0, x0] + xu*yl*_im[μ, y1, x0] + xl*yu*_im[μ, y0, x1] + xl*yl*_im[μ, y1, x1]

@ti.func # fit a parabola
def _fitCenter(y0: ti.f32, y1: ti.f32, y2: ti.f32, y3: ti.f32, y4: ti.f32) -> ti.f32:
    a = (2*y0 - y1 - 2*y2 - y3 + 2*y4) / 14
    b = -0.2*y0 - 0.1*y1 + 0.1*y3 + 0.2*y4
    return -b/a/2

@ti.func # fit a line
def _fitZero(x1: ti.f32, x2: ti.f32, x3: ti.f32, x4: ti.f32, x5: ti.f32, y1: ti.f32, y2: ti.f32, y3: ti.f32, y4: ti.f32, y5: ti.f32) -> ti.f32:
    a = 5
    c = x1 + x2 + x3 + x4 + x5
    d = x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5
    D = a * d - c * c
    e = y1 + y2 + y3 + y4 + y5
    f = x1*y1 + x2*y2 + x3*y3 + x4*y4 + x5*y5
    b = (e * d - c * f) / D
    k = (f * a - e * c) / D
    return -b/k

@ti.func
def _normalize(arr: ti.template(), ζ: ti.i32, d: ti.i32):
    for μ in range(ζ):
        maxx = 0.0
        minx = 999.0
        for t in range(d): # find max and min, serialized
            if (arr[μ, t] > maxx):
                maxx = arr[μ, t]
            if (arr[μ, t] < minx):
                minx = arr[μ, t]
        for t in range(d):
            arr[μ, t] = (arr[μ, t] - minx) * 2 / (maxx - minx) - 1

@ti.func
def _cX(ζ: ti.i32, d: ti.i32): # d = 0(X)|1(Y)
    _normalize(_x, ζ, 2*R)
    # correlate
    for μ, k in ti.ndrange(ζ, 30):
        _cx[μ, k] = 0
    for μ, k, l in ti.ndrange(ζ, 30, 2*R):
        _cx[μ, k] += _x[μ, l] * _x[μ, k - l + L - 16]
    for μ in range(ζ): # find max in correlate and fit
        x = 15
        for t in range(30):
            if _cx[μ, t] > _cx[μ, x]:
                x = t
        _p[μ, d] += (x - 15 + _fitCenter(_cx[μ, x-2], _cx[μ, x-1], _cx[μ, x], _cx[μ, x+1], _cx[μ, x+2])) / 2

@ti.func
def _XY(ζ: ti.i32):
    for μ, t in ti.ndrange(ζ, (5, 2*R+5)): # sample slice x
        _x[μ, t-5] = _BI(μ, t + _p[μ, 0], R+4) + _BI(μ, t + _p[μ, 0], R+5) + _BI(μ, t + _p[μ, 0], R+6)
    _cX(ζ, 0)
    for μ, t in ti.ndrange(ζ, (5, 2*R+5)): # sample slice y
        _x[μ, t-5] = _BI(μ, R+4, t + _p[μ, 1]) + _BI(μ, R+5, t + _p[μ, 1]) + _BI(μ, R+6, t + _p[μ, 1])
    _cX(ζ, 1)

@ti.func
def _profile(ζ: ti.i32):
    for μ, r in ti.ndrange(ζ, Nr):
        _I[μ, r] = 0
    for μ, r, θ in ti.ndrange(ζ, Nr, Nθ): # intensity profile
        x = R+5 + _p[μ, 0] + Fr * r * ti.cos(θ * Fθ)
        y = R+5 + _p[μ, 1] + Fr * r * ti.sin(θ * Fθ)
        _I[μ, r] += _BI(μ, x, y) / Nθ
    _normalize(_I, ζ, Nr)

@ti.func
def _tilde(ζ: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32):
    l = 2 * Nr
    for μ, k in ti.ndrange(ζ, (wl, wr)):
        _Iq[μ, k] = 0
    for μ, k, r in ti.ndrange(ζ, (wl, wr), l):
        _Iq[μ, k] += _I[μ, ti.abs(Nr-r)] * ti.cos(2*π*k*r/l) * (0.5 - 0.5 * ti.cos(2*π*(k-wl)/(wr-wl-1)))
    for μ, r in ti.ndrange(ζ, Nr):
        _I[μ, r] = 0
        _J[μ, r] = 0
    for μ, k, r in ti.ndrange(ζ, (wl, wr), (rf, Nr)):
        _I[μ, r] += _Iq[μ, k] * ti.cos(2*π*k*(r+Nr)/l) / l
        _J[μ, r] += _Iq[μ, k] * ti.sin(2*π*k*(r+Nr)/l) / l

@ti.func
def _ΔΦ(μ: ti.i32, i: ti.i32, z: ti.i32) -> ti.f32:
    s = 0.0
    t = 0.0
    for r in range(Rf, Nr):
        Δ = (ti.atan2(_J[μ, r], _I[μ, r]) - _Φ[i, z, r]) % (2*π)
        if (Δ > π):
            Δ -= 2*π
        w = ti.sqrt(_I[μ, r] ** 2 + _J[μ, r] ** 2) * _A[i, z, r]
        t += w
        s += w * Δ
    return s / t

@ti.func
def __Z(n: ti.i32, m: ti.i32, rf: ti.i32, nz: ti.i32):
    for i, j in ti.ndrange(n, m):
        μ = j * n + i
        z0 = 0
        minχ2 = 999999999.0
        for z in range(nz):
            χ2 = 0.0
            for r in range(rf, Nr):
                χ2 += (_I[μ, r] - _R[i, z, r]) ** 2
            if (χ2 < minχ2):
                minχ2 = χ2
                z0 = z
        _p[μ, 2] = _fitZero(_Z[z0-2], _Z[z0-1], _Z[z0], _Z[z0+1], _Z[z0+2], _ΔΦ(μ, i, z0-2), _ΔΦ(μ, i, z0-1), _ΔΦ(μ, i, z0), _ΔΦ(μ, i, z0+1), _ΔΦ(μ, i, z0+2))

@ti.kernel # XY & profile
def tiProfile(ζ: ti.i32):
    _XY(ζ) # iterate twice
    _XY(ζ)
    _profile(ζ)

@ti.kernel # for ComputeCalibration
def tiTilde(ζ: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32):
    _tilde(ζ, rf, wl, wr)

@ti.kernel
def tiXYZ(n: ti.i32, m: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32, nz: ti.i32):
    _XY(n * m)
    _XY(n * m)
    _profile(n * m)
    _tilde(n * m, rf, wl, wr)
    __Z(n, m, rf, nz)

# load data into taichi scope
# @return number of beads, number of images
def load(beads, imgs, pos=True):
    n = len(beads)
    m = len(imgs)
    maxζ = _im.shape[0]
    if n * m > maxζ:
        raise Exception("Insufficient Capacity")
    im = []
    p = []
    for img in imgs:
        for b in beads:
            x = int(b.x)
            y = int(b.y)
            im.append(img[y-R-5:y+R+5, x-R-5:x+R+5])
            p.append([b.x - x, b.y - y, 0])
    im = np.pad(np.array(im, dtype=np.int32), ((0, maxζ-n*m), (0, 0), (0, 0)))
    p = np.pad(np.array(p, dtype=np.float32), ((0, maxζ-n*m), (0, 0)))
    _im.from_numpy(im)
    _p.from_numpy(p)
    return n, m

"""
Calculate XY Position
@param beads: list of beads
@param imgs: list of 2d array of image data
@return: [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]]]
"""
def XY(beads, imgs):
    n, m = load(beads, imgs)
    tiProfile(n * m) # compute x, y, I in taichi scope
    p = _p.to_numpy()
    I = _I.to_numpy()
    bp = []
    for b in beads:
        bp.append([int(b.x), int(b.y)])
    res = []
    for j in range(m):
        r = []
        for i in range(n):
            b = beads[i]
            b.x = bp[i][0] + p[i][0]
            b.y = bp[i][1] + p[i][1]
            b.profile = I[i]
            r.append([b.x, b.y])
        res.append(r)
    return res

"""
Calculate I and store
@param beads: list of beads
@param imgs: array of 2d array of image data
@param z: z position
"""
def Calibrate(beads, imgs, z):
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
@param rf: forget radius
@param wl: window left end in Fourier space
@param wr: window right end in Fourier space
"""
def ComputeCalibration(beads, rf=10, wl=3, wr=30):
    global Rf, Wl, Wr, Nz
    Rf = rf
    Wl = wl
    Wr = wr
    n = len(beads)
    if (n == 0):
        return
    for b in beads:
        b.Rc = [] # real part
        b.Φc = [] # phase angle
        b.Ac = [] # amplitude
    Nz = len(beads[0].Zc)
    for z in range(Nz):
        I = []
        for i in range(_I.shape[0]):
            if i < n:
                I.append(beads[i].Ic[z])
            else:
                I.append(np.zeros(Nr))
        _I.from_numpy(np.array(I, dtype=np.float32))
        tiTilde(n, rf, wl, wr)
        I = _I.to_numpy()
        J = _J.to_numpy()
        for i in range(n):
            Ii = I[i][Rf:]
            Ji = J[i][Rf:]
            beads[i].Rc.append(Ii)
            beads[i].Φc.append(np.arctan2(Ji, Ii))
            beads[i].Ac.append(np.sqrt(Ii**2 + Ji**2))
    Zc = []
    Φc = []
    Ac = []
    Rc = []
    for b in beads:
        b.Rc = np.array(b.Rc)
        b.Ac = np.array(b.Ac)
        b.Φc = (np.array(b.Φc) + 2*π) % (2*π)
        b.Φc = np.unwrap(b.Φc, axis=0)
        b.Φc = np.unwrap(b.Φc, axis=1)
        Zc = b.Zc
        Φc.append(b.Φc)
        Ac.append(b.Ac)
        Rc.append(b.Rc)
    maxn = _A.shape[0]
    maxz = _Z.shape[0]
    Zc = np.pad(Zc, (0, maxz-Nz))
    Φc = np.pad(Φc, ((0, maxn-n), (0, maxz-Nz), (Rf, 0)))
    Ac = np.pad(Ac, ((0, maxn-n), (0, maxz-Nz), (Rf, 0)))
    Rc = np.pad(Rc, ((0, maxn-n), (0, maxz-Nz), (Rf, 0)))
    _Z.from_numpy(Zc.astype(np.float32))
    _Φ.from_numpy(Φc.astype(np.float32))
    _A.from_numpy(Ac.astype(np.float32))
    _R.from_numpy(Rc.astype(np.float32))

"""
Calculate XYZ Position (cover XY)
@param beads: list of beads
@param imgs: list of 2d array of image data
@return: [[[x1, y1, z1], [x2, y2, z2]], [[x1, y1, z1], [x2, y2, z2]]]
"""
def XYZ(beads, imgs):
    n, m = load(beads, imgs)
    tiXYZ(n, m, Rf, Wl, Wr, Nz)
    p = _p.to_numpy()
    bp = []
    for b in beads:
        bp.append([int(b.x), int(b.y)])
    res = []
    for j in range(m):
        r = []
        for i in range(n):
            b = beads[i]
            b.x = bp[i][0] + p[i][0]
            b.y = bp[i][1] + p[i][1]
            b.z = p[i][2]
            r.append([b.x, b.y, b.z])
        res.append(r)
    return res

"""
Interface for beads
"""
class Bead:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 0
        # self calibration
        self.Ic = [] # Intensity Profiles
        self.Zc = [] # Z values

    def __repr__(self):
        return f"Bead({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"Bead({self.x}, {self.y}, {self.z})"
