import numpy as np
import matplotlib.pyplot as plt
import time
import mm

start = time.time()
for i in range(100):
    img = mm.Get()
    print(np.sum(img))
print(time.time() - start)
