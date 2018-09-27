import os
import vggish_input
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import soundfile as sf
import pickle
from pydub import AudioSegment
from random import randint

dataset_folder_path = '/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/sameCluster/'
new_path = '/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/sameCluster/'
classes = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']


new_tr_set, labels = [], []
for fold in os.listdir(dataset_folder_path):
    for folder in os.listdir(dataset_folder_path + fold):
        #fold, folder = "Cluster3", "Mix-5"        
        for cls in os.listdir(dataset_folder_path + fold + "/" + folder):
            if ".DS_Store" not in cls:
                for filename in os.listdir(dataset_folder_path + fold + "/" + folder + "/" + cls):
                    if ".DS_Store" not in filename:
                        names = filename.split("_+_")
                        encoding = [0] * 20
                        for n in names:
                            encoding[classes.index(n.split("_")[0])] = 1
                        fts = vggish_input.wavfile_to_examples(dataset_folder_path + fold + "/" + folder + "/" + cls + "/" + filename)
                        #for feat in fts:
                        new_tr_set.append(fts)
                        labels.append(encoding)
                        print(names, ":", encoding)
    
        x, x_rest, y, y_rest = train_test_split(new_tr_set, labels, test_size=0.0000001)
        with open(new_path + fold + "/test_" + folder + ".txt", "wb") as fp:
            pickle.dump((x, y), fp)


















