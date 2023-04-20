import numpy as np
import taichi as ti
import time

def SelectBeads(T, getImg, R=35):
    img = getImg()
    W = len(img[0])
    H = len(img)
    beads = []
    gui = ti.GUI('Select Beads', (W, H))
    while gui.running:
        img = getImg()
        T.XY(beads, [img])
        gui.set_image(np.flip(np.transpose(img), axis=1))
        for j in range(len(beads)):
            b = beads[j]
            gui.rect([(b.x-R)/W, 1 - (b.y-R)/H], [(b.x+R)/W, 1 - (b.y+R)/H], color=0xff0000)
            gui.circle([b.x/W, 1 - b.y/H], color=0xff0000)
            gui.text(str(j), [(b.x-R)/W, 1 - (b.y-R)/H], color=0xff0000)
        gui.get_event()
        if gui.is_pressed(ti.GUI.LMB) or gui.is_pressed(ti.GUI.RMB):
            x, y = gui.get_cursor_pos()
            x = x * W
            y = H - y * H
            for b in beads:
                if (b.x-R < x and b.x+R > x and b.y-R < y and b.y+R > y):
                    beads.remove(b)
            if gui.is_pressed(ti.GUI.LMB):
                beads.append(T.Bead(x, y))
        gui.show()
    return beads

# WITHOUT ComputeCalibration!
def Calibrate(beads, T, getImg, getZ, setZ, Nz=100, step=100, m=10, R=35):
    img = getImg()
    W = len(img[0])
    H = len(img)
    gui = ti.GUI('Calibrate', (W, H))
    zcs = np.arange(0, Nz*step, step)
    i = 0
    sz = getZ()
    imgs = []
    while gui.running:
        img = getImg()
        imgs.append(img)
        T.XY(beads, [img])
        gui.set_image(np.flip(np.transpose(img), axis=1))
        gui.line([0, 0], [(i*m+len(imgs))/(Nz*m), 0], radius=6, color=0x0000ff)
        gui.text("z = " + str(zcs[i]), [0.01, 0.05], color=0x0000ff)
        for j in range(len(beads)):
            b = beads[j]
            gui.rect([(b.x-R)/W, 1 - (b.y-R)/H], [(b.x+R)/W, 1 - (b.y+R)/H], color=0xff0000)
            gui.circle([b.x/W, 1 - b.y/H], color=0xff0000)
            gui.text(str(j), [(b.x-R)/W, 1 - (b.y-R)/H], color=0xff0000)
        if len(imgs) == m: # calibrate
            T.Calibrate(beads, imgs, zcs[i])
            i += 1
            if i == Nz: # calibration end
                setZ(sz)
                return beads
            imgs = []
            setZ(sz + zcs[i])
            time.sleep(0.2)
        gui.show()
    setZ(sz)
    return beads

def Track(beads, T, getImg, R=35):
    img = getImg()
    W = len(img[0])
    H = len(img)
    gui = ti.GUI('Tracking', (W, H))
    cot = 0
    trace = []
    for i in range(len(beads)):
        trace.append([])
    while gui.running:
        img = getImg()
        T.XYZ(beads, [img])
        cot += 1
        gui.set_image(np.flip(np.transpose(img), axis=1))
        gui.text("cot = " + str(cot), [0.01, 0.05], color=0x0000ff)
        for j in range(len(beads)):
            b = beads[j]
            trace[j].append([b.x, b.y, b.z])
            gui.rect([(b.x-R)/W, 1 - (b.y-R)/H], [(b.x+R)/W, 1 - (b.y+R)/H], color=0xff0000)
            gui.circle([b.x/W, 1 - b.y/H], color=0xff0000)
            gui.text(str(j), [(b.x-R)/W, 1 - (b.y-R)/H], color=0xff0000)
        gui.show()
    return trace