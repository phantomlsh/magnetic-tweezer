import numpy as np
import utils
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
Find center near (x, y)
@param img: 2d array of image data
@param x: expected bead x position
@param y: expected bead y position
@param it: iteration times
@return: x, y position
"""
def XY(img, x, y, it):
    xl = int(x)
    yl = int(y)
    xline = np.sum(img[yl-2:yl+3, xl-R:xl+R], axis=0)
    yline = np.sum(img[yl-R:yl+R, xl-2:xl+3], axis=1)
    return xl + utils.SymmetryCenter(xline, it), yl + utils.SymmetryCenter(yline, it)

"""
Compute single side intensity profile
@param img: 2d array of image data
@param x: expected bead x position
@param y: expected bead y position
@return: 1d array
"""
def Profile(img, x, y):
    Is = utils.BilinearInterpolate(img, sxs+x, sys+y).reshape((Nθ, Nr))
    return utils.NormalizeArray(np.sum(Is, axis=0))

"""
Compute filtered profile It
@param I: 1d array, single side intensity profile
@param rf: radius of forgetness, in unit of Nr
@return: 1d array
"""
def Tilde(I, rf):
    I = np.append(np.flip(I), I)
    Iq = np.fft.fft(I) * utils.Gaussian(np.fft.fftfreq(I.shape[-1]), 0.05, 0.02)
    It = np.fft.ifft(Iq)
    return It[rf+Nr:len(It)]

"""
Interface for beads
"""
class Bead:
    def __init__(self, x, y, rf=20):
        self.x = x
        self.y = y
        self.rf = rf
        # self calibration
        self.Zc = [] # Z values
        self.Rc = [] # real part
        self.Φc = [] # phase angle
        self.Ac = [] # amplitude

    def __repr__(self):
        return f"Bead({self.x}, {self.y}, rf={self.rf})"

    def __str__(self):
        return f"Bead({self.x}, {self.y}, rf={self.rf})"

    def XY(self, img, it=3):
        self.x, self.y = XY(img, self.x, self.y, it)
        return self.x, self.y

    def Calibrate(self, img, z):
        It = Tilde(Profile(img, self.x, self.y), self.rf)
        self.Zc.append(z)
        self.Rc.append(np.real(It))
        self.Φc.append(np.unwrap(np.angle(It)))
        self.Ac.append(np.abs(It))

    def Z(self, img):
        It = Tilde(Profile(img, self.x, self.y), self.rf)
        Ri = np.real(It)
        Φi = np.unwrap(np.angle(It))
        Ai = np.abs(It)
        χ2 = np.sum((Ri-self.Rc)**2, axis=1)
        mini = np.argmin(χ2)
        ΔΦ = np.average(Φi-self.Φc, axis=1, weights=Ai*self.Ac)
        plt.plot(self.Zc, ΔΦ)
        p = np.polynomial.polynomial.polyfit(self.Zc, ΔΦ, 1)
        self.z = -p[0]/p[1]
        return self.z