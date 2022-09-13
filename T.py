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
def SetParams(r=40, nr=80, nθ=80, maxn=50, maxz=100):
    global R, L, freq, Nr, Nθ, Fr, Fθ
    R = r
    L = r * 2
    freq = np.fft.rfftfreq(L*2)
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ
    global _im, _p, _I, _J, _Z, _Φ, _A, _R, _Iq, _x, _cx
    _im = ti.field(dtype=ti.i32, shape=(maxn, 2*R+10, 2*R+10)) # img grid
    _p = ti.field(dtype=ti.f32, shape=(maxn, 3)) # points
    _I = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # intensity
    _J = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # imaginary part
    _Z = ti.field(dtype=ti.f32, shape=(maxz)) # calibration z
    _Φ = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration angle
    _A = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration amplitude
    _R = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration real
    # caches
    _Iq = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # fourier space
    _x = ti.field(dtype=ti.f32, shape=(maxn, 2*R))
    _cx = ti.field(dtype=ti.f32, shape=(maxn, 30))

SetParams()

@ti.func # Bilinear Interpolate
def _BI(i: ti.i32, x: ti.f32, y: ti.f32) -> ti.f32:
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    xu = x1 - x
    xl = x - x0
    yu = y1 - y
    yl = y - y0
    return xu*yu*_im[i, y0, x0] + xu*yl*_im[i, y1, x0] + xl*yu*_im[i, y0, x1] + xl*yl*_im[i, y1, x1]

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
def _parallelNormalize(arr: ti.template(), n: ti.i32, d: ti.i32):
    for i in range(n):
        _min[i] = 999999.0
        _max[i] = 0.0
    for i, t in ti.ndrange(n, d):
        ti.atomic_max(_max[i], arr[i, t])
        ti.atomic_min(_min[i], arr[i, t])
    for i, t in ti.ndrange(n, d):
        arr[i, t] = (arr[i, t] - _min[i]) * 2 / (_max[i] - _min[i]) - 1

@ti.func
def _normalize(arr: ti.template(), n: ti.i32, d: ti.i32):
    for i in range(n): # find max and min, serialized
        maxx = 0.0
        minx = 999.0
        for t in range(d):
            if (arr[i, t] > maxx):
                maxx = arr[i, t]
            if (arr[i, t] < minx):
                minx = arr[i, t]
        for t in range(d):
            arr[i, t] = (arr[i, t] - minx) * 2 / (maxx - minx) - 1

@ti.func
def _cX(n: ti.i32, d: ti.i32): # d = 0(X)|1(Y)
    _normalize(_x, n, 2*R)
    # correlate
    for i, k in ti.ndrange(n, 30):
        _cx[i, k] = 0
    for i, k, l in ti.ndrange(n, 30, 2*R):
        _cx[i, k] += _x[i, l] * _x[i, k - l + L - 16]
    for i in range(n): # find max in correlate and fit
        x = 15
        for t in range(30):
            if _cx[i, t] > _cx[i, x]:
                x = t
        _p[i, d] += (x - 15 + _fitCenter(_cx[i, x-2], _cx[i, x-1], _cx[i, x], _cx[i, x+1], _cx[i, x+2])) / 2

@ti.func
def _XY(n: ti.i32):
    for i, t in ti.ndrange(n, (5, 2*R+5)): # sample slice x
        _x[i, t-5] = _BI(i, t + _p[i, 0], R+4) + _BI(i, t + _p[i, 0], R+5) + _BI(i, t + _p[i, 0], R+6)
    _cX(n, 0)
    for i, t in ti.ndrange(n, (5, 2*R+5)): # sample slice y
        _x[i, t-5] = _BI(i, R+4, t + _p[i, 1]) + _BI(i, R+5, t + _p[i, 1]) + _BI(i, R+6, t + _p[i, 1])
    _cX(n, 1)

@ti.func
def _profile(n: ti.i32):
    for i, r in ti.ndrange(n, Nr):
        _I[i, r] = 0
    for i, r, θ in ti.ndrange(n, Nr, Nθ): # intensity profile
        x = R+5 + _p[i, 0] + Fr * r * ti.cos(θ * Fθ)
        y = R+5 + _p[i, 1] + Fr * r * ti.sin(θ * Fθ)
        _I[i, r] += _BI(i, x, y) / Nθ
    _normalize(_I, n, Nr)

@ti.func
def _tilde(n: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32):
    l = 2 * Nr
    for i, k in ti.ndrange(n, (wl, wr)):
        _Iq[i, k] = 0
    for i, k, r in ti.ndrange(n, (wl, wr), l):
        _Iq[i, k] += _I[i, ti.abs(Nr-r)] * ti.cos(2*π*k*r/l) * (0.5 - 0.5 * ti.cos(2*π*(k-wl)/(wr-wl-1)))
    for i, r in ti.ndrange(n, Nr):
        _I[i, r] = 0
        _J[i, r] = 0
    for i, k, r in ti.ndrange(n, (wl, wr), (rf, Nr)):
        _I[i, r] += _Iq[i, k] * ti.cos(2*π*k*(r+Nr)/l) / l
        _J[i, r] += _Iq[i, k] * ti.sin(2*π*k*(r+Nr)/l) / l

@ti.func
def _ΔΦ(i: ti.i32, z: ti.i32) -> ti.f32:
    s = 0.0
    t = 0.0
    for r in range(Rf, Nr):
        Δ = (ti.atan2(_J[i, r], _I[i, r]) - _Φ[i, z, r]) % (2*π)
        if (Δ > π):
            Δ -= 2*π
        w = ti.sqrt(_I[i, r] ** 2 + _J[i, r] ** 2) * _A[i, z, r]
        t += w
        s += w * Δ
    return s / t

@ti.func
def __Z(n: ti.i32, rf: ti.i32, nz: ti.i32):
    for i in range(n):
        z0 = 0
        minχ2 = 999999999.0
        for z in range(nz):
            χ2 = 0.0
            for r in range(rf, Nr):
                χ2 += (_I[i, r] - _R[i, z, r]) ** 2
            if (χ2 < minχ2):
                minχ2 = χ2
                z0 = z
        _p[i, 2] = _fitZero(_Z[z0-2], _Z[z0-1], _Z[z0], _Z[z0+1], _Z[z0+2], _ΔΦ(i, z0-2), _ΔΦ(i, z0-1), _ΔΦ(i, z0), _ΔΦ(i, z0+1), _ΔΦ(i, z0+2))

@ti.kernel # XY & profile
def tiProfile(n: ti.i32):
    _XY(n) # iterate twice
    _XY(n)
    _profile(n)

@ti.kernel # for ComputeCalibration
def tiTilde(n: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32):
    _tilde(n, rf, wl, wr)

@ti.kernel
def tiXYZ(n: ti.i32, rf: ti.i32, wl: ti.i32, wr: ti.i32, nz: ti.i32):
    _XY(n)
    _XY(n)
    _profile(n)
    _tilde(n, rf, wl, wr)
    __Z(n, rf, nz)

# load data into taichi scope
# @return number of beads
def load(beads, img, pos=True):
    n = len(beads)
    im = []
    p = []
    for b in beads:
        x = int(b.x)
        y = int(b.y)
        im.append(img[y-R-5:y+R+5, x-R-5:x+R+5])
        p.append([b.x - x, b.y - y, 0])
    maxn = _im.shape[0]
    im = np.pad(np.array(im, dtype=np.int32), ((0, maxn-n), (0, 0), (0, 0)))
    p = np.pad(np.array(p, dtype=np.float32), ((0, maxn-n), (0, 0)))
    _im.from_numpy(im)
    _p.from_numpy(p)
    return n

"""
Calculate XY Position
@param beads: list of beads
@param img: 2d array of image data
"""
def XY(beads, img):
    n = load(beads, img)
    tiProfile(n) # compute x, y, I in taichi scope
    p = _p.to_numpy()
    I = _I.to_numpy()
    for i in range(n):
        b = beads[i]
        b.x = int(b.x) + p[i][0]
        b.y = int(b.y) + p[i][1]
        b.profile = I[i]

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
        XY(beads, img)
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
@param img: 2d array of image data
"""
def XYZ(beads, img):
    n = load(beads, img)
    tiXYZ(n, Rf, Wl, Wr, Nz)
    p = _p.to_numpy()
    for i in range(n):
        b = beads[i]
        b.x = int(b.x) + p[i][0]
        b.y = int(b.y) + p[i][1]
        b.z = int(b.z) + p[i][2]

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
