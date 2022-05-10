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
def SetParams(r=40, nr=80, nθ=80):
    global R, L, freq
    R = r
    L = r * 2
    freq = np.fft.rfftfreq(L*2)
    kernel.SetParams(r, nr, nθ)

SetParams()

# shift from the center
def CenterShift(array, it=2):
    normalized = 2 * array / np.max(array) - 1 # normalize
    fft = np.fft.rfft(np.append(normalized, np.zeros(L)))
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

"""
Calculate XY Position
@param beads: list of beads
@param img: 2d array of image data
@param it: iteration times
"""
def XY(beads, img, it=2):
    for b in beads:
        xl = int(b.x)
        yl = int(b.y)
        xline = np.sum(img[yl-2:yl+3, xl-R:xl+R], axis=0)
        yline = np.sum(img[yl-R:yl+R, xl-2:xl+3], axis=1)
        b.x = xl + CenterShift(xline, it)
        b.y = yl + CenterShift(yline, it)

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
        ΔΦ = np.average(Φi-b.Φc[i-3:i+4], axis=1, weights=Ai*b.Ac[i-3:i+4])
        p = np.polynomial.polynomial.polyfit(b.Zc[i-3:i+4], ΔΦ, 1)
        #plt.scatter(b.Zc, ΔΦ)
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
