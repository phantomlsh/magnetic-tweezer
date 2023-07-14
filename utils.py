import numpy as np
from scipy import signal
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Normalize to [-1, 1]
def NormalizeArray(array):
    return 2 * array / np.max(array) - 1

def Gaussian(x, μ, σ):
    return np.exp(-np.power(x-μ, 2.) / (2 * np.power(σ, 2.)))

def CropImage(img, x, y, L):
    xi = int(x)
    yi = int(y)
    r = int(L/2)
    return img[yi-r:yi+r, xi-r:xi+r]

def HoughCircles(image, minRadius, maxRadius):
    img = (image / image.max() * 255).astype("uint8")
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if (circles is None):
        return []
    return circles[0]

def PlotCalibration(bead):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
    z = len(bead.Ic)
    r = len(bead.Ic[0])
    ax0.set_title("Profile")
    ax0.set_ylabel("z (stack)")
    ax0.set_xlabel("r (sample distance)")
    ax0.imshow(np.flip(bead.Ic, axis=0), cmap="gray", extent=[0, r, 0, z])
    ax1.set_title("Real Part")
    ax1.set_xlabel("r (sample distance)")
    ax1.imshow(np.flip(bead.Rc, axis=0), cmap="gray", extent=[0, r, 0, z])
    ax2.set_title("Phase")
    ax2.set_xlabel("r (sample distance)")
    ax2.imshow(np.flip(bead.Φc, axis=0), extent=[0, r, 0, z])
    plt.show()

def PlotXY(trace, s=1, show=True):
    t = np.transpose(np.array(trace))
    x = t[0] - np.mean(t[0])
    y = t[1] - np.mean(t[1])
    r = max(np.abs(x).max(), np.abs(y).max())    
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    ax = fig.add_subplot(gs[1:4, 0:3])
    ax_x = fig.add_subplot(gs[0, 0:3])
    ax_y = fig.add_subplot(gs[1:4, 3])
    ax.scatter(x, y, s)
    ax.grid()
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax_x.hist(x, bins=30)
    ax_y.hist(y, bins=30, orientation="horizontal")
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_yticklabels(), visible=False)
    mean_value = np.mean(t[0])
    std_value = np.std(t[0])
    label_text = "X(px) Mean: %.2f nm, Std Deviation: %.2f nm" % (mean_value * 166, std_value * 166)
    ax.set_xlabel(label_text)
    mean_valu = np.mean(t[1])
    std_valu = np.std(t[1])
    label_tex = "Y(px) Mean: %.2f nm, Std Deviation: %.2f nm" % (mean_value * 166, std_value * 166)
    ax.set_ylabel(label_tex)
    ax_y.set_xlabel("Count")
    ax_x.set_ylabel("Count")
    if show:
        plt.show()
def PlotX(trace):
    t = np.transpose(np.array(trace))
    x = t[0]
    plt.plot(x)
    mean_value = np.mean(t[0])
    std_value = np.std(t[0])
    label_text = "X(px) Mean: %.2f, Std Deviation: %.2f" % (mean_value, std_value)
    plt.xlabel('500 times with FPS about 25')
    plt.ylabel(label_text)
    plt.title('X Position')
    plt.grid()
    plt.show()
def PlotY(trace):
    t = np.transpose(np.array(trace))
    y = t[1]
    plt.plot(y)
    mean_value = np.mean(t[1])
    std_value = np.std(t[1])
    label_text = "Y(px) Mean: %.2f, Std Deviation: %.2f" % (mean_value, std_value)
    plt.xlabel('500 times with FPS about 25')
    plt.ylabel(label_text)
    plt.title('Y Position')
    plt.grid()
    plt.show()
def TraceAxis(trace, axis=2):
    t = np.transpose(np.array(trace))
    return t[axis]
