import T
import UI
import mm
import utils
import matplotlib.pyplot as plt

beads = UI.SelectBeads(T, mm.Get)
print(beads)

UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ)
T.ComputeCalibration(beads)
utils.PlotCalibration(beads[0])
mm.SetZ(mm.GetZ() + 5000)
trace = UI.Track(beads, T, mm.Get)
utils.PlotXY(trace[0])

plt.plot(utils.TraceAxis(trace[0]) - utils.TraceAxis(trace[1]))
plt.xlabel("Z(nm)")
plt.title("Z Position")
plt.grid()
plt.show()
