import numpy as np
import matplotlib.pyplot as plt
import time, mm, utils
import N as T
import UI

z0 = mm.GetZ()

beads = UI.SelectBeads(T, mm.Get)

UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ)

for i in range(5, len(beads)):
    beads[i].rf = 14 # reference beads
T.ComputeCalibration(beads)

for i in range(len(beads)):
    utils.PlotCalibration(beads[i])

δz = 4000
mm.SetZ(z0 + δz)

trace = UI.Track(beads, T, mm.Get, 500)
Δt = trace[0] - trace[1]
utils.PlotXY(Δt)
plt.plot(utils.TraceAxis(Δt))
plt.xlabel("Z(nm)")
plt.title("Z Position")
plt.grid()
plt.show()

"""
# test
zts = np.arange(500, 9500, 50)
z0s = []
for z in zts:
    mm.SetZ(sz + z)
    time.sleep(0.2)
    for t in range(100):
        img = mm.Get()
        T.XYZ(beads, [img])
        z0s.append(beads[0].z)

xs = []
ys = []
yerr = []
for i, z in enumerate(zts):
    x = sz + z
    data = z0s[i*100:(i*100+100)]
    xs.append(x)
    ys.append(np.mean(data) - x)
    yerr.append(np.std(data))

utils.PlotCalibration(beads[0])

plt.grid()
plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
plt.title('Bias in Z tracking')
plt.xlabel('Z(nm)')
plt.ylabel('Bias(nm)')
plt.show()

mm.SetZ(sz)
"""