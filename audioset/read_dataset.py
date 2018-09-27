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

dataset_folder_path = '/cs/scratch/pmh20/ExtraWork/AugmentedDataset/TrainingSet/'
new_path = '/cs/scratch/pmh20/ExtraWork/AugmentedDataset/BinaryTrSet-PP/'
classes = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']


for cl in classes:	
    new_tr_set, not_class, labels, nc_labels = [], [], [], []
    for folder in os.listdir(dataset_folder_path):
        if folder == cl:
            encoding = [1, 0]
        else:
            encoding = [0, 1]
        if ".DS_Store" not in folder:
            for filename in os.listdir(dataset_folder_path + folder):
                if ".DS_Store" not in filename:
                    fts = vggish_input.wavfile_to_examples(dataset_folder_path + folder + "/" + filename)
                    if folder == cl:    
                        for feat in fts:
                            new_tr_set.append(feat)
                            labels.append(encoding)
                    else:
                        for feat in fts:
                            not_class.append(feat)
                            nc_labels.append(encoding)
                   
    nc_to_add, x_rest, ncl_to_add, y_rest = train_test_split(not_class, nc_labels, test_size=0.95)
    print(len(new_tr_set), ":", len(nc_to_add))
    for i, x in enumerate(nc_to_add):
        new_tr_set.append(x)
        labels.append(ncl_to_add[i])
    train_val_split = train_test_split(new_tr_set, labels, test_size=0.15)
    with open(new_path + cl + ".txt", "wb") as fp:
        pickle.dump(train_val_split, fp)

    
        
    
    
    



















