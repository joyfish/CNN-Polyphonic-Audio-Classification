from __future__ import print_function
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def get_mel(y, sr):
    librosa.feature.melspectrogram(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    librosa.frames_to_time(onset_frames, sr=sr)

    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    D = librosa.stft(y)

    plt.figure()
    plt.plot(y)
    plt.show()

    plt.figure()
    fr = [abs(x) for x in np.fft.fft(y)]
    plt.plot(fr[:len(fr)/2])
    # plt.plot([sum(x) for x in zip(*D)])
    plt.show()


    # ax1 = plt.subplot(2, 1, 1)
    # # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log')
    # plt.title('Power spectrogram')
    # plt.subplot(2, 1, 2, sharex=ax1)
    # plt.plot(y[:4000])
    # plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    # plt.axis('tight')
    # plt.legend(frameon=True, framealpha=0.75)
    # plt.show()

    # Using a pre-computed power spectrogram
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    S2 = S

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S2 = librosa.feature.mfcc(S=librosa.power_to_db(S2))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfccs, x_axis='time')
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()

    return mfccs

y, sr = librosa.load('Dataset/Laughing/257924__erikschenkel__laughing-man-5.wav')
get_mel(y, sr)