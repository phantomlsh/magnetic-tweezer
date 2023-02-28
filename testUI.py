import T as T
import UI
import mm
import utils

beads = UI.SelectBeads(T, mm.Get)
print(beads)

UI.Calibrate(beads, T, mm.Get, mm.GetZ, mm.SetZ)
T.ComputeCalibration(beads)
utils.PlotCalibration(beads[0])
