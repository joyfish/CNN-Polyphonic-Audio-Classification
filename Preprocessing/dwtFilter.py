from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pywt
import numpy as np
import soundfile as sf
import seaborn
from statsmodels.robust import mad


def noise_removal_1(x):
    data, samplerate = sf.read(x)
    audio = (x[1][:, 0] + x[1][:, 1]) / 2

    cA, cD = pywt.dwt(audio, 'haar')
    cAt = pywt.threshold(cA, np.std(cA)/1)
    cDt = pywt.threshold(cD, np.std(cD)/1)
    dwtInv = pywt.idwt(cAt, cDt, 'haar')
    dwtInv = [int(i) for i in dwtInv]
    dwtInv = np.asarray(dwtInv, dtype='int16')
    print(audio[:100])
    print(dwtInv[:100])
    sf.write('noiseOutput.wav', dwtInv, samplerate, subtype='PCM_16')
    sf.write('noiseOutput1.wav', audio, samplerate, subtype='PCM_16')

    remainingSound = audio - dwtInv
    sf.write('noiseOutput2.wav', remainingSound, samplerate, subtype='PCM_16')

    plt.subplot(3, 1, 1)
    plt.plot(audio[:4000])
    plt.subplot(3, 1, 2)
    plt.plot(dwtInv[:4000])
    plt.subplot(3, 1, 3)
    plt.plot(remainingSound[:4000])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.show()
    
    
def noise_removal_2(x, wavelet="db4", level=1, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold
    sigma = mad(coeff[-level])
    # univeral threshold
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold( i, value=uthresh, mode="soft") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec(coeff, wavelet, mode="per")
    f, ax = plt.subplots()
    plot(x, color="b", alpha=0.5)
    plot(y, color="b")
    if title:
        ax.set_title(title)
    ax.set_xlim((0,len(y)))
