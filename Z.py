import numpy as np
from utils import Gaussian

Rf = 20
Nr = 80

# @return I tilde
def Tilde(I):
  I = np.append(np.flip(I), I)
  Iq = np.fft.fft(I) * Gaussian(np.fft.fftfreq(I.shape[-1]), 0.05, 0.018)
  It = np.fft.ifft(Iq)
  return It[Rf+Nr:len(It)]
