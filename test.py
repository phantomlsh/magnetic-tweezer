import time
import T, UI
import numpy as np
import mm, utils, acs
import matplotlib.pyplot as plt

acs.To(39)
z0 = mm.GetZ()

beads = UI.SelectBeads(T, mm.Get)

UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ)

for i in range(1, len(beads)):
    beads[i].rf = 14 # reference beads
T.ComputeCalibration(beads)

for i in range(len(beads)):
    utils.PlotCalibration(beads[i])

δz = 4000
mm.SetZ(z0 + δz)

trace = UI.Track(beads, T, mm.Get, 500)
utils.PlotXY(trace[0])
plt.plot(utils.TraceAxis(trace[0]) - utils.TraceAxis(trace[1]))
plt.xlabel("Z(nm)")
plt.title("Z Position")
plt.grid()
plt.show()

data = []
for magneticHeight in np.arange(39, 25, -0.5):
    acs.To(magneticHeight)
    mm.SetZ(z0 + δz)
    print(magneticHeight)
    time.sleep(1)
    mm.Get()
    trace = UI.Track(beads, T, mm.Get, 500)
    data.append(trace)
allTrace = []
for h in range(len(data)):
    trace = data[h]
    for j in range(len(trace[0])):
        one = []
        for i in range(len(trace)):
            one.append(trace[i][j][0])
            one.append(trace[i][j][1])
            one.append(trace[i][j][2])
        allTrace.append(one)
allTrace = np.array(allTrace)
acs.To(39)

plt.plot((allTrace[:,2] - allTrace[:,5])[:])
plt.show()

f = open("./data/trace3.dat", "w")
for t in allTrace:
    for d in t:
        f.write(str(d) + ",")
    f.write("\n")
f.close()
