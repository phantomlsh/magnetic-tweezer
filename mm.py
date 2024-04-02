"""
MicroManager Interface
for image and Z position

2022.07.06 Phantomlsh
"""

from pycromanager import Core
import atexit
import numpy as np
import time

core = Core()
core.get_version_info()

# Snap an image to get height and width
core.snap_image()
tagged_image = core.get_tagged_image()
Height = tagged_image.tags["Height"]
Width = tagged_image.tags["Width"]
print("Mi(Py)croManager Initializing...")
print("Acquisition Size: Width =", Width, " Height =", Height)

core.start_continuous_sequence_acquisition(1)
time.sleep(1)


def exit_handler():
    core.stop_sequence_acquisition()
    print("Mi(Py)croManager Exiting...")


atexit.register(exit_handler)


# Acquire image
def Get():
    image = core.get_last_image()
    return np.reshape(image, (Height, Width))


def GetZ():
    return (
        float(core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)")) * 1000
    )


print("Z =", GetZ())


def SetZ(z):
    core.set_property("MCL NanoDrive Z Stage", "Set position Z (um)", z / 1000)
