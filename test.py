import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T

beads = []

circles = utils.HoughCircles(mm.Get(), 15, 30)
for c in circles:
    if (c[0] > 80 and c[0] < 680 and c[1] > 80 and c[1] < 500):
        beads.append(T.Bead(c[0], c[1]))

n = len(beads)
print(n, "Beads:", beads)

T.test(beads, mm.Get())
