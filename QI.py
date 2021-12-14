import numpy as np
from math import floor, pi as π
from numba import jit
from utils import BilinearInterpolate, SymmetryCenter

# Nθ should be a multiple of 4
def SamplePoints(N, Nθ, Nr):
	rs = np.tile(np.arange(0, N/2, N/2/Nr), Nθ)
	θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
	return rs * np.cos(θs), rs * np.sin(θs)

@jit(nopython=True, fastmath=True, cache=True)
def Quarters(Is, Nθ):
	q = floor(Nθ/4)
	return np.sum(Is[0:q], axis=0), np.sum(Is[q:2*q], axis=0), np.sum(Is[2*q:3*q], axis=0), np.sum(Is[3*q:4*q], axis=0)

def XY(Is, Nθ):
	Itr, Itl, Ibl, Ibr = Quarters(Is, Nθ)
	Il = (Itl + Ibl) / 2
	Ir = (Itr + Ibr) / 2
	It = (Itl + Itr) / 2
	Ib = (Ibl + Ibr) / 2
	Ix = np.append(np.flip(Il), Ir)
	Iy = np.append(np.flip(Ib), It)
	return SymmetryCenter(Ix), SymmetryCenter(Iy)
