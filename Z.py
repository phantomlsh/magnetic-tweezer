import numpy as np
from utils import Gaussian
import matplotlib.pyplot as plt

# @params nr: sampling number in radial direction
def Init(nr):
	global Nr, Zc, Rc, Φc, Ac
	Zc = [] # Z values of calibration
	Rc = [] # real part
	Φc = [] # phase angle
	Ac = [] # amplitude
	Nr = nr

# @params I: Intensity Profile (single side)
# @params rf: Forget Radius
# @return I tilde
def Tilde(I, rf=20):
	I = np.append(np.flip(I), I)
	Iq = np.fft.fft(I) * Gaussian(np.fft.fftfreq(I.shape[-1]), 0.05, 0.02)
	It = np.fft.ifft(Iq)
	return It[rf+Nr:len(It)]

# @params z: current z value
# @params It: I tilde
def Calibrate(z, It):
	Zc.append(z)
	Rc.append(np.real(It))
	Φc.append(np.unwrap(np.angle(It)))
	Ac.append(np.abs(It))

def Z(It):
	Ri = np.real(It)
	Φi = np.unwrap(np.angle(It))
	Ai = np.abs(It)
	χ2 = np.sum((Ri-Rc)**2, axis=1)
	mini = np.argmin(χ2)
	ΔΦ = np.average(Φi-Φc, axis=1, weights=Ai*Ac)
	plt.plot(Zc, ΔΦ)
	p = np.polynomial.polynomial.polyfit(Zc, ΔΦ, 1)
	return -p[0]/p[1]
