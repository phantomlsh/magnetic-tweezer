import numpy as np

π = np.pi

"""
Global params initialization
@param r: expected bead radius in pixel
@param nθ: sampling number in polar direction
@param nr: sampling number in radial direction
"""
def SetParams(r=35, nr=80, nθ=80):
    global R, L, freq, Nr, Nθ, sxs, sys
    R = r
    L = r * 2
    Nr = nr
    Nθ = nθ
    freq = np.fft.rfftfreq(L*2)
    rs = np.tile(np.arange(0, R, R/nr), nθ)
    θs = np.repeat(np.arange(π/nθ, 2*π, 2*π/nθ), nr)
    # relative sample points
    sxs = rs * np.cos(θs)
    sys = rs * np.sin(θs)

SetParams()

# shift from the center
def centerShift(array, it=2):
    array = (array - np.mean(array)) / np.std(array)
    fft = np.fft.rfft(np.append(array, np.zeros(L)))
    res = 0
    d = 0
    for loop in range(it):
        fft *= np.exp(2j*π*d*freq)
        co = np.fft.irfft(fft**2)[R-1:L+R-1]
        i = np.argmax(co[R-30:R+30]) + R-30
        p = np.polynomial.polynomial.polyfit(np.arange(i-2, i+3), co[i-2:i+3], 2)
        d = -p[1]/4/p[2] - R/2
        res += d
    return res

# Cannot deal with boundary, avoid boundary
def bilinearInterpolate(im, x, y):
    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    xu = x1 - x
    xl = x - x0
    yu = y1 - y
    yl = y - y0
    return xu*yu*im[y0, x0] + xu*yl*im[y1, x0] + xl*yu*im[y0, x1] + xl*yl*im[y1, x1]

def profile(beads, img):
    for b in beads:
        b.profile = np.average(bilinearInterpolate(img, sxs + b.x, sys + b.y).reshape((Nθ, Nr)), axis=0)
        b.profile = (b.profile - np.mean(b.profile)) / np.std(b.profile)

def tilde(I, rf, w):
    I = np.append(np.flip(I), I)
    Iq = np.fft.fft(I)
    l = len(Iq)
    win = np.append(np.zeros(w[0]), np.hanning(w[1]-w[0]))
    win = np.append(win, np.zeros(l-w[1]))
    It = np.fft.ifft(Iq*win)
    return It[rf+(len(It)//2):len(It)]

"""
Calculate XY Position
@param beads: list of beads
@param imgs: list of 2d array of image data
@param it: iteration times
@return: [Bead0 Trace, Bead1 Trace, ...]
"""
def XY(beads, imgs, it=2):
    res = []
    for b in beads:
        res.append([])
    for img in imgs:
        for i in range(len(beads)):
            b = beads[i]
            xl = int(b.x)
            yl = int(b.y)
            xline = np.sum(img[yl-2:yl+3, xl-R:xl+R], axis=0)
            yline = np.sum(img[yl-R:yl+R, xl-2:xl+3], axis=1)
            b.x = xl + centerShift(xline, it)
            b.y = yl + centerShift(yline, it)
            res[i].append([b.x, b.y])
    return res


"""
Calculate I and store
@param beads: list of beads
@param imgs: list of 2d array of image data
@param z: z position
"""
def Calibrate(beads, imgs, z):
    for b in beads:
        b.l = []
    for img in imgs:
        XY(beads, [img])
        profile(beads, img)
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
    for b in beads:
        b.Rc = [] # real part
        b.Φc = [] # phase angle
        b.Ac = [] # amplitude
        for I in b.Ic:
            It = tilde(I, b.rf, b.w)
            b.Rc.append(np.real(It))
            b.Φc.append(np.angle(It))
            b.Ac.append(np.abs(It))

"""
Calculate XYZ Position
@param beads: list of beads
@param imgs: list of 2d array of image data
@return: [Bead0 Trace, Bead1 Trace, ...]
"""
def XYZ(beads, imgs):
    res = []
    for b in beads:
        res.append([])
    for img in imgs:
        XY(beads, [img])
        profile(beads, img)
        for i in range(len(beads)):
            b = beads[i]
            It = tilde(b.profile, b.rf, b.w)
            Ri = np.real(It)
            Φi = np.unwrap(np.angle(It))
            Ai = np.abs(It)
            χ2 = np.sum((Ri-b.Rc)**2, axis=1)
            x = np.argmin(χ2)
            ΔΦ = (Φi-b.Φc[x-3:x+4]) % (2*π)
            np.subtract(ΔΦ, 2*π, out=ΔΦ, where=ΔΦ>π)
            ΔΦ = np.average(ΔΦ, axis=1, weights=Ai*b.Ac[x-3:x+4])
            p = np.polynomial.polynomial.polyfit(b.Zc[x-3:x+4], ΔΦ, 1)
            b.z = -p[0]/p[1]
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
        self.Ic = [] # Intensity Profiles
        self.Zc = [] # Z values

    def __repr__(self):
        return f"Bead({self.x}, {self.y}, {self.z}, rf={self.rf}, w=[{self.w[0]}, {self.w[1]}])"

    def __str__(self):
        return f"Bead({self.x}, {self.y}, {self.z}, rf={self.rf}, w=[{self.w[0]}, {self.w[1]}])"
