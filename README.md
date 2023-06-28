# magnetic-tweezer

A Python library to track beads in magnetic tweezer experiments

## Get started

```python
import N as T # Numpy version
# OR
import T as T # Taichi version
```

Create beads

```python
b = T.Bead(123, 456) # x, y position
beads = [T.Bead(100, 120), b] # put beads into a list
```

Track XY position

```python
T.XY(beads, [img]) # img is numpy 2d array

print(beads)
print(beads[0].x, beads[0].y)
```

Calibrate for Z position

```python
imgs = [img1, img2, img3]
z = 10
# add one calibration point:
T.Calibrate(beads, imgs, z)

# add more calibration points...

# complete calibration
T.ComputeCalibration(beads)
```

Track Z position (also XY position)

**Must have called `ComputeCalibration` for the same set of beads**

```python
T.XYZ(beads, [img])
print(beads)
```

When using GPU acceleration, you can pass multiple images into parallel processing

```python
res = T.XYZ(beads, [img1, img2, img3])
print(res)
"""
[   # XYZ tracking result
	[   # bead1 trace
		[100, 120, 122.5], # bead1 in img1
		[200, 22.2, 12.5], # bead1 in img2
	],
	[   # bead2 trace
		[101, 120, 122.9], # bead2 in img1
		[202, 20.2, 10.2], # bead2 in img2
	],
]
"""
```

## Directory

```python
# acs.py: control magnetic motor
acs.To(30) # unit in mm

# mm.py: mi(py)cromanager interface
mm.Get() # get last image
mm.GetZ() # get piezo position in nm
mm.SetZ(1000) # set piezo position in nm

# N.py: CPU version tracking with NumPy
# T.py: GPU version tracking with Taichi

# UI.py: useful UI utilities with Taichi
beads = UI.SelectBeads(T, mm.Get, R=35) # select beads with mouse clicking
UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ, Nz=100, step=100, m=10, R=35) # automatic calibration with UI for observation
UI.Track(beads, T, mm.Get, maxCot=-1, R=35) # track beads with UI for observation, LOW performance!

# utils.py: useful utilities in image processing and plotting
```
