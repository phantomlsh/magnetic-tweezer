import numpy as np
import matplotlib.pyplot as plt
from utils import BilinearInterpolate, Gaussian, SymmetryCenter
import simulate
import QI

img = np.ndarray((100, 100))
x = 50
y = 50

N = 80
Nθ = 80
Nr = 80
xs, ys = QI.SamplePoints(N, Nθ, Nr)

simulate.Generate(50, 50, 50, 4, img)
I = BilinearInterpolate(img, xs+x, ys+y)
I = I.reshape((Nθ, Nr))
Δx, Δy = QI.XY(I, Nθ)

print(Δx, Δy)
