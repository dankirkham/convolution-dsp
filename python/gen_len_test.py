#!/usr/bin/env python3
import numpy as np
from scipy.signal import oaconvolve
import time

kernel = np.float32(np.random.rand(51))

re = np.float32(np.random.rand(100))
im = np.float32(np.random.rand(100))
signal = np.array(list(map(lambda v: v[0] + v[1] * 1j, zip(re, im))))
signal = np.complex64(signal)

output = oaconvolve(signal, kernel, "same")
print(len(output))
assert len(output) == 100
output = np.complex64(output)

with open("kernel_odd.bin", "wb") as f:
    kernel.tofile(f)

with open("signal_odd.bin", "wb") as f:
    signal.tofile(f)

with open("output_odd.bin", "wb") as f:
    output.tofile(f)

kernel = np.float32(np.random.rand(50))

re = np.float32(np.random.rand(100))
im = np.float32(np.random.rand(100))
signal = np.array(list(map(lambda v: v[0] + v[1] * 1j, zip(re, im))))
signal = np.complex64(signal)

output = oaconvolve(signal, kernel, "same")
print(len(output))
assert len(output) == 100
output = np.complex64(output)

with open("kernel_even.bin", "wb") as f:
    kernel.tofile(f)

with open("signal_even.bin", "wb") as f:
    signal.tofile(f)

with open("output_even.bin", "wb") as f:
    output.tofile(f)
