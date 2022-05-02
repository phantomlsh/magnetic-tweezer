import numpy as np
import utils
import kernel
import matplotlib.pyplot as plt

π = np.pi

"""
Global params initialization
@param r: expected bead radius in pixel
@param nθ: sampling number in polar direction
@param nr: sampling number in radial direction
"""
def Init(r=40, nθ=80, nr=80):
    global Nθ, Nr, R, sxs, sys
    Nθ = nθ
    Nr = nr
    R = r
    rs = np.tile(np.arange(0, R, R/Nr), Nθ)
    θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
    # relative sample points
    sxs = rs * np.cos(θs)
    sys = rs * np.sin(θs)

"""
Calculate XY Position
@param beads: list of beads
@param img: 2d array of image data
@param it: iteration times
"""
def XY(beads, img, it=3):
    for b in beads:
        xl = int(b.x)
        yl = int(b.y)
        xline = np.sum(img[yl-2:yl+3, xl-R:xl+R], axis=0)
        yline = np.sum(img[yl-R:yl+R, xl-2:xl+3], axis=1)
        b.x = xl + utils.SymmetryCenter(xline, it)
        b.y = yl + utils.SymmetryCenter(yline, it)

"""
Calculate I and store
@param beads: list of beads
@param img: 2d array of image data
@param it: iteration times
"""
def Calibrate(imgs, beads, z):
    for img in imgs:
        XY(img, beads)
        for b in beads:
            l = []
            for img in imgs:
                I = Profile(img, self.x, self.y)
                l.append(I)
            self.Ic.append(np.average(l, axis=0))
            self.Zc.append(z)

"""
Compute single side intensity profile
@param img: 2d array of image data
@param x: expected bead x position
@param y: expected bead y position
@return: 1d array
"""
def Profile(img, x, y):
    n = Nθ*Nr
    res = np.zeros(n)
    im = img.astype(int)
    kernel.tiBI(im, n, sxs+x, sys+y, res)
    Is = res.reshape((Nθ, Nr))
    return utils.NormalizeArray(np.sum(Is, axis=0))

"""
Compute filtered profile It
@param I: 1d array, single side intensity profile
@param rf: radius of forgetness, in unit of Nr
@param w: window range, in unit of Nr
@return: 1d array
"""
def Tilde(I, rf, w):
    I = np.append(np.flip(I), I)
    Iq = np.fft.fft(I)
    q = np.fft.fftfreq(I.shape[-1])
    L = len(Iq)
    win = np.append(np.zeros(w[0]), np.hanning(w[1]-w[0]))
    win = np.append(win, np.zeros(L-w[1]))
    It = np.fft.ifft(Iq*win)
    return It[rf+Nr:len(It)]

"""
Interface for beads
"""
class Bead:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self calibration
        self.Ic = [] # Intensity Profiles
        self.Zc = [] # Z values

    def __repr__(self):
        return f"Bead({self.x}, {self.y})"

    def __str__(self):
        return f"Bead({self.x}, {self.y})"

    def XY(self, img, it=2):
        self.x, self.y = XY(img, self.x, self.y, it)
        return self.x, self.y

    def Calibrate(self, imgs, z):
        l = []
        for img in imgs:
            self.XY(img)
            I = Profile(img, self.x, self.y)
            l.append(I)
        self.Ic.append(np.average(l, axis=0))
        self.Zc.append(z)

    def ComputeCalibration(self, rf=12, w=[5, 15]):
        self.rf = rf
        self.w = w
        self.Rc = [] # real part
        self.Φc = [] # phase angle
        self.Ac = [] # amplitude
        for I in self.Ic:
            It = Tilde(I, rf, w)
            self.Rc.append(np.real(It))
            self.Φc.append(np.unwrap(np.angle(It)))
            self.Ac.append(np.abs(It))

    def Z(self, img):
        It = Tilde(Profile(img, self.x, self.y), self.rf, self.w)
        Ri = np.real(It)
        Φi = np.unwrap(np.angle(It))
        Ai = np.abs(It)
        χ2 = np.sum((Ri-self.Rc)**2, axis=1)
        i = np.argmin(χ2)
        ΔΦ = np.average(Φi-self.Φc, axis=1, weights=Ai*self.Ac)
        p = np.polynomial.polynomial.polyfit(self.Zc[i-6:i+6], ΔΦ[i-6:i+6], 1)
        plt.scatter(self.Zc, ΔΦ)
        self.z = -p[0]/p[1]     
        return self.z