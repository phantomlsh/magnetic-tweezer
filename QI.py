import numpy as np
from math import floor, pi as π
from utils import SymmetryCenter, NormalizeArray

# Nθ should be a multiple of 4
def SamplePoints(L, Nθ, Nr):
	rs = np.tile(np.arange(0, L/2, L/2/Nr), Nθ)
	θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
	return rs * np.cos(θs), rs * np.sin(θs)

def XY(Is, Nθ):
	q = floor(Nθ/4)
	Itr = np.sum(Is[0:q], axis=0)
	Itl = np.sum(Is[q:2*q], axis=0)
	Ibl = np.sum(Is[2*q:3*q], axis=0)
	Ibr = np.sum(Is[3*q:4*q], axis=0)
	Il = (Itl + Ibl) / 2
	Ir = (Itr + Ibr) / 2
	It = (Itl + Itr) / 2
	Ib = (Ibl + Ibr) / 2
	Ix = np.append(np.flip(Il), Ir)
	Iy = np.append(np.flip(Ib), It)
	return SymmetryCenter(Ix), SymmetryCenter(Iy)

def Profile(Is):
	return NormalizeArray(np.sum(Is, axis=0))
