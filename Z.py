import numpy as np
from utils import Gaussian

def Init(nr, rf):
	global Nr, Rf, Zc, Rc, Φc, Ac
	Zc = [] # Z value of calibration
	Rc = [] # real part
	Φc = [] # phase angle
	Ac = [] # amplitude
	Nr = nr
	Rf = rf # Forget Radius

# @return I tilde
def Tilde(I):
	I = np.append(np.flip(I), I)
	Iq = np.fft.fft(I) * Gaussian(np.fft.fftfreq(I.shape[-1]), 0.05, 0.018)
	It = np.fft.ifft(Iq)
	return It[Rf+Nr:len(It)]
