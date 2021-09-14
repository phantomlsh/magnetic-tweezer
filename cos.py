import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-20, 20, 0.5)
y0 = np.exp(-0.1*np.abs(x))
y1 = np.cos(x)*np.exp(-0.1*np.abs(x))
y2 = np.cos(x-0.25)*np.exp(-0.1*np.abs(x-0.25))
y3 = np.cos(x-0.5)*np.exp(-0.1*np.abs(x-0.5))

#co0 = np.fft.ifft(np.fft.fft(y0)*np.conjugate(np.fft.fft(np.flip(y0))))
# co11 = np.fft.ifft(np.conjugate(np.fft.fft(y1))*np.fft.fft(np.flip(y1)))
# co12 = np.fft.ifft(np.fft.fft(y1)*np.conjugate(np.fft.fft(np.flip(y1))))

co21 = np.fft.ifft(np.conjugate(np.fft.fft(y2))*np.fft.fft(np.flip(y2)))
co22 = np.fft.ifft(np.fft.fft(y2)*np.conjugate(np.fft.fft(np.flip(y2))))
co3 = np.fft.ifft(np.conjugate(np.fft.fft(y3))*np.fft.fft(np.flip(y3)))

# plt.plot(y1)
# plt.plot(y2)
# plt.plot(y3)

plt.plot(np.abs(co21))
plt.plot(np.abs(co22))

#plt.plot(np.arctan(np.imag(co2)/np.real(co2)))
#plt.plot(np.arctan(np.imag(co3)/np.real(co3)))
plt.show()
