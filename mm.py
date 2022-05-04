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

def Get():
    image = core.get_last_image()
    return np.reshape(image, (Height, Width))

def GetZ():
    return float(core.get_property("MCL NanoDrive Z Stage", "Set position Z (um)")) * 1000

print("Z =", GetZ())

def SetZ(z):
    core.set_property("MCL NanoDrive Z Stage", "Set position Z (um)", z / 1000)
