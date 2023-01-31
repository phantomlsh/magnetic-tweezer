import numpy as np
import taichi as ti

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
        for b in beads:
            gui.rect([(b.x-R)/W, 1 - (b.y-R)/H], [(b.x+R)/W, 1 - (b.y+R)/H], color=0xff0000)
            gui.circle([b.x/W, 1 - b.y/H], color=0xff0000)
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