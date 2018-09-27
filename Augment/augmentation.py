import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from random import randint

n = 0

def load_audio_file(file_path):
    data = librosa.core.load(file_path)[0]
    return data


def plot_time_series(data):
    plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


def white_noise(fname, path, magnification, new_name):
    d, samplerate = sf.read(fname)
    try:
        data = d[:, 0] + d[:, 1]
        data = [d/2 for d in data]
    except: data = d
    wn = np.random.randn(len(data))
    mag = 0.005*magnification
    noise = [n*mag for n in wn]
    data_wn = data + noise
    sf.write(path + new_name, data_wn, int(samplerate/2), subtype='PCM_16')


def pitch_change(fname, path, magnification, up, new_name):
    sound = AudioSegment.from_file(fname, format="wav")
    # shift the pitch up by half an octave (speed will increase proportionally)
    octaves = 0.25*magnification
    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    new_sample_rate2 = int(sound.frame_rate / (2.0 ** octaves))
    # keep the same samples but tell the computer they ought to be played at the
    # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
    pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    pitch_sound2 = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate2})
    # convert it to a common sample rate (44.1k - standard audio CD)
    pitch_sound = pitch_sound.set_frame_rate(44100)
    pitch_sound2 = pitch_sound2.set_frame_rate(44100)
    # export / save pitch changed sound
    if up:
        pitch_sound.export(path + new_name, format="wav")
    else:
        pitch_sound2.export(path + new_name, format="wav")


def volume_change(fname, path, magnification, up, new_name):
    sound = AudioSegment.from_file(fname, format="wav")
    if up:
        sound_vol = sound + 5*magnification
    else:
        sound_vol = sound - 5*magnification
    sound_vol.export(path + new_name, format="wav")


dataset_folder_path = os.getcwd() + '/ValidationSet/'
new_path = os.getcwd() + '/AugmentedDataset/ValidationSet/'
largestFolder, n = 0., 0

for folder in os.listdir(dataset_folder_path):
    if ".DS_Store" not in folder:
        n = 0
        size = sum(os.path.getsize(dataset_folder_path + folder + "/" + f) for f in os.listdir(dataset_folder_path + folder) if os.path.isfile(dataset_folder_path + folder + "/" + f))
        for filename in os.listdir(dataset_folder_path + folder):
            print(dataset_folder_path + folder + "/" + filename)
            
            if ".DS_Store" not in filename:
                try:
                    sound = AudioSegment.from_file(dataset_folder_path + folder + "/" + filename, format="wav")
                    sound.export(new_path + folder + "/" + folder + "_" + str(n) + ".wav", format="wav")
                except: 1    
                print(size)
                n += 1
                if size > largestFolder:
                    largestFolder = size

for folder in os.listdir(dataset_folder_path):
    if ".DS_Store" not in folder:
        n = 0
        print(folder, largestFolder)
        while True:
            size = sum(os.path.getsize(new_path + folder + "/" + f) for f in os.listdir(new_path + folder) if
                       os.path.isfile(new_path + folder + "/" + f))
            if size >= largestFolder:
                break

            for filename in os.listdir(dataset_folder_path + folder):
                if ".DS_Store" not in filename:
                    try:
                        size = sum(os.path.getsize(new_path + folder + "/" + f) for f in os.listdir(new_path + folder) if os.path.isfile(new_path + folder + "/" + f))
                        print(size)
                        if size < largestFolder:
                            aug_choice = randint(0, 4)
                            magnification_choice = randint(1, 2)
                            n += 1
                        else:
                            break
                        name = dataset_folder_path + folder + "/" + filename
                        if aug_choice == 0:
                            white_noise(name, new_path + folder + "/", magnification_choice, folder + "_" + str(n) + ".wav")
                        if aug_choice == 1:
                            pitch_change(name, new_path + folder + "/", magnification_choice, True, folder + "_" + str(n) + ".wav")
                        if aug_choice == 2:
                            volume_change(name, new_path + folder + "/", magnification_choice, True, folder + "_" + str(n) + ".wav")
                        if aug_choice == 3:
                            pitch_change(name, new_path + folder + "/", magnification_choice, False, folder + "_" + str(n) + ".wav")
                        if aug_choice == 4:
                            volume_change(name, new_path + folder + "/", magnification_choice, False, folder + "_" + str(n) + ".wav")
                    except: 1


