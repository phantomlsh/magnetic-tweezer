import numpy as np
from math import floor, pi as π
from utils import SymmetryCenter, NormalizeArray

# Nθ should be a multiple of 4
def SamplePoints(L, Nθ, Nr):
	rs = np.tile(np.arange(0, L/2, L/2/Nr), Nθ)
	θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
	return rs * np.cos(θs), rs * np.sin(θs)

def XY(Is, Nθ):
	q = int(Nθ/4)
	hq = int(q/2)
	Itr = np.sum(Is[0:q], axis=0)
	Itl = np.sum(Is[q:2*q], axis=0)
	Ibl = np.sum(Is[2*q:3*q], axis=0)
	Ibr = np.sum(Is[3*q:4*q], axis=0)
	Il = np.sum(Is[3*hq:5*hq], axis=0)
	Ir = np.sum(Is[0:hq], axis=0) + np.sum(Is[7*hq:8*hq], axis=0)
	It = np.sum(Is[hq:3*hq], axis=0)
	Ib = np.sum(Is[5*hq:7*hq], axis=0)
	Ix = np.append(np.flip(Il), Ir)
	Iy = np.append(np.flip(Ib), It)
	return SymmetryCenter(Ix), SymmetryCenter(Iy)

def Profile(Is):
	return NormalizeArray(np.sum(Is, axis=0))
