import numpy as np
import utils
import Z
import matplotlib.pyplot as plt

π = np.pi

# @param r: expected bead radius in pixel
# @param nθ: sampling number in polar direction
# @param nr: sampling number in radial direction
def Init(r=40, nθ=80, nr=80):
	global Nθ, Nr, L, R, sxs, sys
	Nθ = nθ
	Nr = nr
	R = r
	rs = np.tile(np.arange(0, R, R/Nr), Nθ)
	θs = np.repeat(np.arange(π/Nθ, 2*π, 2*π/Nθ), Nr)
	# relative sample points
	sxs = rs * np.cos(θs)
	sys = rs * np.sin(θs)
	Z.Init(Nr)

# Find center near (x, y)
# @return (x, y)
def XY(img, x, y, r=40, it=3):
	xl = int(x)
	yl = int(y)
	xline = np.sum(img[yl-2:yl+3, xl-r:xl+r], axis=0)
	yline = np.sum(img[yl-r:yl+r, xl-2:xl+3], axis=1)
	return xl + utils.SymmetryCenter(xline, it), yl + utils.SymmetryCenter(yline, it)

# @return Array[Nr]
def Profile(img, x, y):
	Is = utils.BilinearInterpolate(img, sxs+x, sys+y)
	Is = Is.reshape((Nθ, Nr))
	Is = utils.NormalizeArray(np.sum(Is, axis=0))
	return Is

# @params snap: function to get current image
# @params get: function to get current Z value in nm
# @params set: function to set Z value in nm
# @params nz: sampling number in Z direction
# @params step: sampling step length in nm
def Calibrate(snap, get, set, x, y, nz=20, step=50):
	sz = get()
	for i in range(nz):
		z = get()
		img = snap()
		x, y = XY(img, x, y)
		I = Profile(img, x, y)
		It = Z.Tilde(I)
		Z.Calibrate(z, It)
		set(z + step)
	set(sz)
	