
Rf = 10

I = QI.Profile(I)
I = np.append(np.flip(I), I)
Iq = np.fft.fft(I) * Gaussian(np.fft.fftfreq(I.shape[-1]), 0, 0.05)
Ix = np.fft.ifft(Iq)
print(Ix)
plt.plot(np.real(Ix))
plt.plot(I)
plt.show()
