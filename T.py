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
def SetParams(r=40, nr=80, nθ=80, maxn=30, maxz=100):
    global R, L, freq, Nr, Nθ, Fr, Fθ
    R = r
    L = r * 2
    freq = np.fft.rfftfreq(L*2)
    R = r
    Nr = nr
    Nθ = nθ
    Fr = R/Nr
    Fθ = 2*π/Nθ
    global _im, _p, _I, _J, _Z, _Φ, _A, _Iq, _x, _y, _cx, _cy
    _im = ti.field(dtype=ti.i32, shape=(1000, 1000)) # img data
    _p = ti.Vector.field(dtype=ti.f32, n=3, shape=(maxn)) # points
    _I = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # intensity
    _J = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # imaginary part
    _Z = ti.field(dtype=ti.f32, shape=(maxz)) # calibration z
    _Φ = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration angle
    _A = ti.field(dtype=ti.f32, shape=(maxn, maxz, Nr)) # calibration amplitude
    # caches
    _Iq = ti.field(dtype=ti.f32, shape=(maxn, Nr)) # fourier space
    _x = ti.field(dtype=ti.f32, shape=(maxn, 2*R))
    _y = ti.field(dtype=ti.f32, shape=(maxn, 2*R))
    _cx = ti.field(dtype=ti.f32, shape=(maxn, 30))
    _cy = ti.field(dtype=ti.f32, shape=(maxn, 30))

SetParams()

@ti.func
def tiBI(x: ti.f32, y: ti.f32) -> ti.f32:
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    xu = x1 - x
    xl = x - x0
    yu = y1 - y
    yl = y - y0
    return xu*yu*_im[y0, x0] + xu*yl*_im[y1, x0] + xl*yu*_im[y0, x1] + xl*yl*_im[y1, x1]

@ti.func
def tiFitCenter(y0: ti.f32, y1: ti.f32, y2: ti.f32, y3: ti.f32, y4: ti.f32) -> ti.f32:
    a = (2*y0 - y1 - 2*y2 - y3 + 2*y4) / 14
    b = -0.2*y0 - 0.1*y1 + 0.1*y3 + 0.2*y4
    return -b/a/2

@ti.func
def tiXY(n: int):
    for i, t in ti.ndrange(n, (-R, R)): # sample slice
        x = _p[i][0]
        y = _p[i][1]
        _x[i, t+R] = tiBI(x + t, y - 1) + tiBI(x + t, y) + tiBI(x + t, y + 1)
        _y[i, t+R] = tiBI(x - 1, y + t) + tiBI(x, y + t) + tiBI(x + 1, y + t)
    for i in range(n): # find max and min, serialized
        maxx = 0.0
        maxy = 0.0
        minx = 999.0
        miny = 999.0
        for t in range(2*R):
            if (_x[i, t] > maxx):
                maxx = _x[i, t]
            if (_y[i, t] > maxy):
                maxy = _y[i, t]
            if (_x[i, t] < minx):
                minx = _x[i, t]
            if (_y[i, t] < miny):
                miny = _y[i, t]
        for t in range(2*R):
            _x[i, t] = (_x[i, t] - minx) * 2 / (maxx - minx) - 1
            _y[i, t] = (_y[i, t] - miny) * 2 / (maxy - miny) - 1
    # correlate
    for i, k in ti.ndrange(n, 30):
        _cx[i, k] = 0
        _cy[i, k] = 0
    for i, k, l in ti.ndrange(n, 30, 2*R):
        _cx[i, k] += _x[i, l] * _x[i, k - l + L - 16]
        _cy[i, k] += _y[i, l] * _y[i, k - l + L - 16]
    for i in range(n): # find max in correlate and fit
        x = 15
        y = 15
        for t in range(30):
            if _cx[i, t] > _cx[i, x]:
                x = t
            if _cy[i, t] > _cy[i, y]:
                y = t
        _p[i][0] += (x - 15 + tiFitCenter(_cx[i, x-2], _cx[i, x-1], _cx[i, x], _cx[i, x+1], _cx[i, x+2])) / 2
        _p[i][1] += (y - 15 + tiFitCenter(_cy[i, y-2], _cy[i, y-1], _cy[i, y], _cy[i, y+1], _cy[i, y+2])) / 2

@ti.kernel
def tiProfile(n: int):
    tiXY(n)
    tiXY(n)
    for i, r in ti.ndrange(n, Nr):
        _I[i, r] = 0
    for i, r, θ in ti.ndrange(n, Nr, Nθ): # intensity profile
        x = _p[i][0] + Fr * r * ti.cos(θ * Fθ)
        y = _p[i][1] + Fr * r * ti.sin(θ * Fθ)
        _I[i, r] += tiBI(x, y) / Nθ
    for i in range(n): # find max and min, serialized
        maxI = 0.0
        minI = 999.0
        for r in range(Nr):
            if (_I[i, r] > maxI):
                maxI = _I[i, r]
            if (_I[i, r] < minI):
                minI = _I[i, r]
        for r in range(Nr): # normalize intensity profile
            _I[i, r] = (_I[i, r] - minI) * 2 / (maxI - minI) - 1

@ti.func
def _tilde(n: int):
    l = 2 * Nr
    for i, k in ti.ndrange(n, (Wl, Wr)):
        _Iq[i, k] = 0
    for i, k, r in ti.ndrange(n, (Wl, Wr), l):
        _Iq[i, k] += _I[i, ti.abs(Nr-r)] * ti.cos(2*π*k*r/l) * (0.5 - 0.5 * ti.cos(2*π*(k-Wl)/(Wr-Wl-1)))
    for i, r in ti.ndrange(n, Nr):
        _I[i, r] = 0
        _J[i, r] = 0
    for i, k, r in ti.ndrange(n, (Wl, Wr), (Rf, Nr)):
        _I[i, r] += _Iq[i, k] * ti.cos(2*π*k*(r+Nr)/l) / l
        _J[i, r] += _Iq[i, k] * ti.sin(2*π*k*(r+Nr)/l) / l

@ti.func
def tiΔΦ(i: int, z: int) -> ti.f32:
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

@ti.kernel
def tiTilde(n: int):
    _tilde(n)

@ti.kernel
def tiZ(n: int):
    _tilde(n)
    for i in range(n):
        z0 = 0
        minχ2 = 999999999.0
        for z in range(Nz):
            χ2 = 0.0
            for r in range(Rf, Nr):
                χ2 += (_I[i, r] - _A[i, z, r] * ti.cos(_Φ[i, z, r])) ** 2
            if (χ2 < minχ2):
                minχ2 = χ2
                z0 = z
        tiΔΦ(i, z0)
        _p[i][2] = _Z[z0]


"""
Calculate XY Position
@param beads: list of beads
@param img: 2d array of image data
"""
def XY(beads, img):
    global _im
    if (_im.shape[0] != img.shape[0] or _im.shape[1] != img.shape[1]):
        _im = ti.field(dtype=ti.i32, shape=(img.shape[0], img.shape[1]))
    _im.from_numpy(img.astype(np.int32))
    n = len(beads)
    for i in range(n):
        _p[i][0] = beads[i].x
        _p[i][1] = beads[i].y
    tiProfile(n)
    p = _p.to_numpy()
    I = _I.to_numpy()
    for i in range(n):
        beads[i].x = p[i][0]
        beads[i].y = p[i][1]
        beads[i].profile = I[i]

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
def ComputeCalibration(beads, rf=12, wl=5, wr=15):
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
        tiTilde(n)
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
    for b in beads:
        b.Rc = np.array(b.Rc)
        b.Ac = np.array(b.Ac)
        b.Φc = (np.array(b.Φc) + 2*π) % (2*π)
        b.Φc = np.unwrap(b.Φc, axis=0)
        b.Φc = np.unwrap(b.Φc, axis=1)
        Zc = b.Zc
        Φc.append(b.Φc)
        Ac.append(b.Ac)
    maxn = _A.shape[0]
    maxz = _Z.shape[0]
    Zc = np.pad(Zc, (0, maxz-Nz))
    Φc = np.pad(Φc, ((0, maxn-n), (0, maxz-Nz), (Rf, 0)))
    Ac = np.pad(Ac, ((0, maxn-n), (0, maxz-Nz), (Rf, 0)))
    _Z.from_numpy(Zc.astype(np.float32))
    _Φ.from_numpy(Φc.astype(np.float32))
    _A.from_numpy(Ac.astype(np.float32))

"""
Calculate Z Position
@param beads: list of beads
@param img: useless
"""
def Z(beads, img=[]):
    n = len(beads)
    tiZ(n)
    p = _p.to_numpy()
    for i in range(n):
        beads[i].z = p[i][2]
    return
    for b in beads:
        It = tilde(b.profile)
        Ri = np.real(It)
        Φi = np.unwrap((np.angle(It) + (2*π)) % (2*π))
        Ai = np.abs(It)
        χ2 = np.sum((Ri-b.Rc)**2, axis=1)
        i = np.argmin(χ2)
        # while Φi[5] - b.Φc[i][5] > 3:
        #     Φi = Φi - 2*π
        # while Φi[5] - b.Φc[i][5] < -3:
        #     Φi = Φi + 2*π
        ΔΦ = np.average(Φi-b.Φc[i-3:i+4], axis=1, weights=Ai*b.Ac[i-3:i+4])
        p = np.polynomial.polynomial.polyfit(b.Zc[i-3:i+4], ΔΦ, 1)
        b.z = -p[0]/p[1]

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
