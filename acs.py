"""
ACS Motion Control Interface
for Z axis motor

2022.07.06 Phantomlsh
"""

MIN = 0
MAX = 39

import ctypes
io = ctypes.windll.LoadLibrary("ACSCL_x64.dll")
double = ctypes.c_double

# connect to ACS Motion Control
# Note the encoding of string
hc = io.acsc_OpenCommEthernetTCP(b"10.0.0.100", 701)

if (hc == -1):
    print("Fail to connect ACS Motion Control")
else:
    io.acsc_Enable(hc, 0, 0)
    print("Connected to ACS Motion Control")

def To(pos):
    if pos < MIN or pos > MAX:
        return
    io.acsc_ToPoint(hc, 0, 0, double(pos), 0)
