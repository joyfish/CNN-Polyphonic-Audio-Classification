from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pywt
import numpy as np
import soundfile as sf

data, samplerate = sf.read('dw.wav')
# sf.write('dw.wav', data, samplerate, subtype='PCM_16')

# read audio samples
input = read("cm.wav")
input2 = read("dw.wav")
input3 = read("cmDw.wav")
audio = (input[1][:, 0] + input[1][:, 1]) / 2
audio2 = (input2[1][:, 0] + input2[1][:, 1]) / 2
audio3 = (input3[1][:, 0] + input3[1][:, 1]) / 2


cA, cD = pywt.dwt(audio3, 'haar')
cAt = pywt.threshold(cA, np.std(cA)/1)
cDt = pywt.threshold(cD, np.std(cD)/1)
dwtInv = pywt.idwt(cAt, cDt, 'haar')
dwtInv = [int(i) for i in dwtInv]
dwtInv = np.asarray(dwtInv, dtype='int16')
print(audio3[:100])
print(dwtInv[:100])
sf.write('noiseOutput.wav', dwtInv, samplerate, subtype='PCM_16')
sf.write('noiseOutput1.wav', audio3, samplerate, subtype='PCM_16')


remainingSound = audio3 - dwtInv
sf.write('noiseOutput2.wav', remainingSound, samplerate, subtype='PCM_16')


plt.subplot(3, 1, 1)
plt.plot(audio3[:4000])
plt.subplot(3, 1, 2)
plt.plot(dwtInv[:4000])
plt.subplot(3, 1, 3)
plt.plot(remainingSound[:4000])
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()
