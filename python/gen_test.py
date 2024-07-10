#!/usr/bin/env python3
import numpy as np
from scipy.signal import oaconvolve
import time

kernel = np.float32(np.random.rand(4001))

re = np.float32(np.random.rand(2**15))
im = np.float32(np.random.rand(2**15))
signal = np.array(list(map(lambda v: v[0] + v[1] * 1j, zip(re, im))))
signal = np.complex64(signal)

for _ in range(10):
    t1 = time.time()
    output = oaconvolve(kernel, signal)
    t2 = time.time()
    print(f"Convolution took {t2 - t1} seconds")
output = np.complex64(output)

with open("kernel.bin", "wb") as f:
    kernel.tofile(f)

with open("signal.bin", "wb") as f:
    signal.tofile(f)

with open("output.bin", "wb") as f:
    output.tofile(f)

print(signal[10])
