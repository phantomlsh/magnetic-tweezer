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
@param imgs: array of 2d array of image data
@param z: z position
"""
def Calibrate(beads, imgs, z):
    for b in beads:
        b.l = []
    for img in imgs:
        XY(beads, img)
        kernel.Profile(beads, img)
        for b in beads:
            b.l.append(b.profile)
    for b in beads:
        b.Ic.append(np.average(b.l, axis=0))
        b.Zc.append(z)

def ComputeCalibration(beads, rf=12, w=[5, 15]):
    for b in beads:
        b.rf = rf
        b.w = w
        b.Rc = [] # real part
        b.Φc = [] # phase angle
        b.Ac = [] # amplitude
        for I in b.Ic:
            It = utils.Tilde(I, rf, w)
            b.Rc.append(np.real(It))
            b.Φc.append(np.angle(It))
            b.Ac.append(np.abs(It))
        b.Φc = np.unwrap(b.Φc, axis=0)
        b.Φc = np.unwrap(b.Φc, axis=1)

def Z(beads, img):
    kernel.Profile(beads, img)
    for b in beads:
        It = utils.Tilde(b.profile, b.rf, b.w)
        Ri = np.real(It)
        Φi = np.unwrap(np.angle(It))
        Ai = np.abs(It)
        χ2 = np.sum((Ri-b.Rc)**2, axis=1)
        i = np.argmin(χ2)
        while Φi[5] - b.Φc[i][5] > 3:
            Φi = Φi - 2*π
        while Φi[5] - b.Φc[i][5] < -3:
            Φi = Φi + 2*π
        ΔΦ = np.average(Φi-b.Φc, axis=1, weights=Ai*b.Ac)
        p = np.polynomial.polynomial.polyfit(b.Zc[i-3:i+4], ΔΦ[i-3:i+4], 1)
        #plt.scatter(b.Zc, ΔΦ)
        b.z = -p[0]/p[1]

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
        return f"Bead({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return f"Bead({self.x}, {self.y}, {self.z})"
