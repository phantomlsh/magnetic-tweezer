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
beads = [T.Bead(100, 120), b]
```

Track XY position

```python
T.XY(beads, img) # img is numpy 2d array

print(beads)
print(beads[0].x, beads[0].y)
```

Further document coming soon...
