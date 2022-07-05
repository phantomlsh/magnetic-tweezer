# magnetic-tweezer

## Get started

```python
# Numpy version
import N as T

# or

# Taichi version
import T as T
```

Create beads

```python
b = T.Bead(123, 456) # x, y position
beads = [T.Bead(100, 120), b] # put beads into a list
```

Track XY position

```python
T.XY(beads, img) # img is numpy 2d array

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
T.XYZ(beads, img)
print(beads)
```
