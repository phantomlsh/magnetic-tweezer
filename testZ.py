import numpy as np
import matplotlib.pyplot as plt
import time
import mm
import utils
import T as T
import UI

z0 = mm.GetZ()

beads = UI.SelectBeads(T, mm.Get)

UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ)

for i in range(0, len(beads)):
    beads[i].rf = 20  # reference beads
T.ComputeCalibration(beads)

# for i in range(len(beads)):
#     utils.PlotCalibration(beads[i])

δz = 2000
mm.SetZ(z0 + δz)

trace = UI.Track(beads, T, mm.Get, 1000)
Δt = trace[0] - trace[1]
t = utils.TraceAxis(Δt)
print(np.std(t))
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [5, 1]})
ax1.set_title("Relative Z Position (1000 frames)")
ax1.plot(t, marker=".")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Z (nm)")
ax1.grid()
ax2.hist(t, bins=30, orientation="horizontal")
ax2.set_xlabel("Count")
plt.show()

# test bias
zts = np.arange(500, 9500, 50)
z0s = []
for z in zts:
    mm.SetZ(z0 + z)
    time.sleep(0.3)
    for t in range(100):
        img = mm.Get()
        T.XYZ(beads, [img])
        z0s.append(beads[0].z)

xs = []
ys = []
yerr = []
for i, z in enumerate(zts):
    data = z0s[i * 100 : (i * 100 + 100)]
    xs.append(z)
    ys.append(np.mean(data) - z0 - z)
    yerr.append(np.std(data))

utils.PlotCalibration(beads[0])

plt.grid()
plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
plt.title("Bias in Z tracking")
plt.xlabel("Z (nm)")
plt.ylabel("Bias (nm)")
plt.show()

mm.SetZ(z0)
