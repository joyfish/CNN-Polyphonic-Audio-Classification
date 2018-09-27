import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

sr, cm = wavfile.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/cm.wav')
sr2, dw = wavfile.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/dw.wav')
sr3, bg = wavfile.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/bg.wav')

dw1 = dw.T[0]  # this is a two channel soundtrack, I get the first track
dw2 = [(x/2**16.0)*2-1 for x in dw1]  # this is 8-bit track, b is now normalized on [-1,1)
fft_dw = fft(dw2)  # calculate fourier transform (complex numbers list)
len_dw = len(fft_dw)/2  # you only need half of the fft list (real signal symmetry)

cm1 = cm.T[0]  # this is a two channel soundtrack, I get the first track
cm2 = [(x/2**16.0)*2-1 for x in cm1]  # this is 8-bit track, b is now normalized on [-1,1)
fft_cm = fft(cm2)  # calculate fourier transform (complex numbers list)
len_cm = len(fft_cm)/2  # you only need half of the fft list (real signal symmetry)

# bg2 = [(x/2**16.0)*2-1 for x in bg]  # this is 8-bit track, b is now normalized on [-1,1)
# fft_bg = fft(bg2)  # calculate fourier transform (complex numbers list)
# len_bg = len(fft_bg)/2  # you only need half of the fft list (real signal symmetry)

plt.plot(abs(fft_dw[:(len_dw-1)]), 'r', label='dw')
plt.plot(abs(fft_cm[:(len_cm-1)]), 'b', label='cm')
# plt.plot(abs(fft_bg[:(len_bg-1)]), 'g', label='bg')
plt.grid(True)
plt.xlim(xmax=20000)
plt.ylim(ymax=80000)
plt.xlabel('Frequency (Hz)')
plt.legend(loc='best')
plt.show()
