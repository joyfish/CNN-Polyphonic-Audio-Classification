import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.decomposition import FastICA
from scipy.io import wavfile
from scipy import linalg


def fast_ICA():
    data, sampleRate = sf.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/cmDw.wav')
    cm, sr1 = sf.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/cm.wav')
    dw, sr2 = sf.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/dw.wav')
    cm = ((cm[:, 0] + cm[:, 1]) / 2)
    dw = ((dw[:, 0] + dw[:, 1]) / 2)

    ica = FastICA(n_components=2)
    s = ica.fit_transform(data)
    a = ica.mixing_
    print(np.mean(a[:, 0]))
    s = s * np.mean(a[:, 0])
    s1 = s[:, 0]
    s2 = ((data[:, 0] + data[:, 1]) / 2) - s1
    sA = s[:, 1]
    sB = ((data[:, 0] + data[:, 1]) / 2) - sA
    s1 = s1 * np.mean(a[:, 0])
    sA = sA * np.mean(a[:, 0])
    print(s[:20])

    sf.write('icaOutput.wav', s, sampleRate, subtype='PCM_16')
    sf.write('1.wav', s1, sampleRate, subtype='PCM_16')
    sf.write('2.wav', s2, sampleRate, subtype='PCM_16')
    sf.write('a.wav', sA, sampleRate, subtype='PCM_16')
    sf.write('b.wav', sB, sampleRate, subtype='PCM_16')

    plt.subplot(3, 1, 1)
    plt.plot(data[:4000])
    plt.subplot(3, 1, 2)
    plt.plot(s[:4000, 0])
    plt.plot(cm[:4000])
    plt.subplot(3, 1, 3)
    plt.plot(s[:4000, 1])
    plt.plot(dw[:4000])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.show()


# Fourth order blind identification (github.com/ShashShukla/ICA/)
def FOBI():
    sampleRate, data = wavfile.read('/Users/peterhaubrick/Desktop/Uni/Project/Blind Source Separation/Sounds/cmDw.wav')
    print(np.amax(data))
    data = data * 1.0 / np.amax(abs(data))
    plt.plot(data)
    plt._show()


def main():
    # fast_ICA()
    FOBI()


if __name__ == "__main__":
    main()
