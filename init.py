from pycromanager import Bridge
import numpy as np

bridge = Bridge()
Core = bridge.get_core()
Core.get_version_info()

Core.snap_image()
tagged_image = Core.get_tagged_image()
Height = tagged_image.tags["Height"]
Width = tagged_image.tags["Width"]
print("Initialization: Snap an image for Width & Height")
print(Width, Height)

def Normalize(image):
	image_array = np.reshape(image, newshape=[-1, Height, Width])
	image_array = (image_array / image_array.max() * 255).astype("uint8")
	return image_array[0,:,:]
