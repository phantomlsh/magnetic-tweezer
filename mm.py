from pycromanager import Bridge
import atexit
import numpy as np
import time

bridge = Bridge()
core = bridge.get_core()
core.get_version_info()

core.snap_image()
tagged_image = core.get_tagged_image()
Height = tagged_image.tags["Height"]
Width = tagged_image.tags["Width"]
print("Initializing...")
print("Width =", Width, " Height =", Height)

core.start_continuous_sequence_acquisition(17)
time.sleep(1)

def exit_handler():
	core.stop_sequence_acquisition()
	print("Exiting...")

atexit.register(exit_handler)

def normalize(image):
	image_array = np.reshape(image, newshape=[-1, Height, Width])
	image_array = (image_array / image_array.max() * 255).astype("uint8")
	return image_array[0,:,:]

def Get():
	image = core.get_last_image()
	image_array = np.reshape(image, newshape=[-1, Height, Width])
	image_array = (image_array / image_array.max() * 255).astype("uint8")
	return image_array[0,:,:]

def GetZ():
	return core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)")

print("Z =", GetZ())

def SetZ(z):
	core.set_property("MCL NanoDrive Z Stage", "Set position Z (um)", z)
