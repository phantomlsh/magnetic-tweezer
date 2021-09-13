from pycromanager import Bridge

bridge = Bridge()
Core = bridge.get_core()
Core.get_version_info()

Core.snap_image()
tagged_image = Core.get_tagged_image()
Height = tagged_image.tags["Height"]
Width = tagged_image.tags["Width"]
print("Initialization: Snap an image for Width & Height")
print(Width, Height)
