import numpy as np
import soundfile as sf
import vggish_input
from sklearn.decomposition import FastICA
import soundfile as sf
import scipy.signal as signal


def fast_ICA(fullPath, numberOfEvents):
    data, sampleRate = sf.read(fullPath)
    ica = FastICA(n_components=numberOfEvents)
    try:
        s = ica.fit_transform(data, numberOfEvents)
    except:
        d = [data, data]
        d2 = np.array(d).T.tolist()
        s = ica.fit_transform(d2, numberOfEvents)
    m = ica.mixing_
    s = s * np.mean(m[:, 0])
    newS = s.T
    a = vggish_input.waveform_to_examples(newS[0], sampleRate)
    b = vggish_input.waveform_to_examples(newS[1], sampleRate)
    unmixedEvents = (a, b)
    return unmixedEvents

    # unmixedEvents = []
    # for i, x in enumerate(newS):
    #     unmixedEvents.append(vggish_input.waveform_to_examples(x, sampleRate))
    # return unmixedEvents


def high_low_pass_filters(fullPath):
    y, sr = sf.read(fullPath)
    filter_stop_freq = 70  # Hz
    filter_pass_freq = 100  # Hz
    filter_order = 1001
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    filtered_audio = signal.filtfilt(filter_coefs, [1], y, axis=0)
    filtered_audio2 = y - filtered_audio
    a = vggish_input.waveform_to_examples(filtered_audio, sr)
    b = vggish_input.waveform_to_examples(filtered_audio2, sr)
    unmixedEvents = (a, b)
    return unmixedEvents
