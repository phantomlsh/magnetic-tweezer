# magnetic-tweezer

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
