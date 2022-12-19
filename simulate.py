from math import sqrt, cos, floor, pi as π
import random

def Generate(X, Y, Z, λ, img):
  L = 100
  for x in range(X-L, X+L):
    for y in range(Y-L, Y+L):
      r = sqrt((x-X)**2 + (y-Y)**2 + Z**2)
      n = (r-Z)/λ
      ϕ = 2*π*(n - floor(n) - 0.5)
      img[y][x] = 5e5 * cos(ϕ/2) * Z**2 / (r**4)
      #img[y][x] = floor(5e5 * cos(ϕ/2) * Z**2 / (r**4) * random.uniform(1, 1.1))
