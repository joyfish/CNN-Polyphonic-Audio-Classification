import numpy as np
import wave
import sys
import math
import contextlib
import soundfile as sf
import scipy.signal as signal


fname = 'cmDw.wav'
outname = 'filtered3.wav'

y, sr = sf.read(fname)
sf.write(fname, y, sr, subtype='PCM_16')


filter_stop_freq = 70  # Hz
filter_pass_freq = 100  # Hz
filter_order = 1001
# High-pass filter
nyquist_rate = sr / 2.
desired = (0, 0, 1, 1)
bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
# Apply high-pass filter
filtered_audio = signal.filtfilt(filter_coefs, [1], y)
sf.write(outname, filtered_audio, sr, subtype='PCM_16')
