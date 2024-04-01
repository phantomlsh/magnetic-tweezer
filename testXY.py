import numpy as np
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
    time.sleep(0.02)
    ts.append(time.time() - start)
    T.XY(beads, [img])
    for i in range(n):
        b = beads[i]
        traces[i].append([b.x, b.y])

traces = np.array(traces)
print("time:", time.time() - start)
Δt = traces[0] - traces[1]
print("X STD =", np.std(utils.TraceAxis(Δt, 0)))
print("Y STD =", np.std(utils.TraceAxis(Δt, 1)))
utils.PlotXY(Δt)
