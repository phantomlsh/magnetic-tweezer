import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import T as T
import UI

beads = []
traces = []

beads = UI.SelectBeads(T, mm.Get)
print(beads)
n = len(beads)
for i in range(n):
    traces.append([])

T.XY(beads, [mm.Get()])

ts = []
start = time.time()
for loop in range(1000):
    img = mm.Get()
    ts.append(time.time() - start)
    T.XY(beads, [img])
    for i in range(n):
        b = beads[i]
        traces[i].append([b.x, b.y])

traces = np.array(traces)
print('time:', time.time() - start)
utils.PlotXY(traces[0] - traces[1])
