import FeatureExtraction as fe
import librosa
import numpy as np
import os


datasetFolderPath = '/Users/peterhaubrick/Desktop/Uni/Project/Feature Extraction/Dataset/'
labels, smallest, averageClasses = {}, [], []
distnaceMatrix = np.zeros((20, 20), dtype=float)
averageCounter = np.zeros((20, 20), dtype=int)

# for folder in os.listdir(datasetFolderPath):
#     small = 0
#     for filename in os.listdir(datasetFolderPath + folder):
#         y, sr = librosa.load(datasetFolderPath + folder + '/' + filename)
#         if np.size(fe.get_mel(y, sr), 1) < small or small == 0:
#             small = np.size(fe.get_mel(y, sr), 1)
#     smallest.append(small)
# print(smallest)
#

y, sr = librosa.load(datasetFolderPath + 'car/175837__amthomas101__car-driving-past-[AudioTrimmer.com].wav')
averages = [87, 33, 33, 207, 96, 33, 305, 39, 22, 37, 25, 139, 107, 46, 96, 131, 333, 206, 18, 87]

for i, folder in enumerate(os.listdir(datasetFolderPath)):
    print(folder)
    classMatrix = np.zeros((np.size(fe.get_mel(y, sr), 0), averages[i]), dtype=float)
    for filename in os.listdir(datasetFolderPath + folder):
        y, sr = librosa.load(datasetFolderPath + folder + '/' + filename)
        classMatrix += fe.get_mel(y, sr)[:, :averages[i]]
    if folder == 'MilkSteamer':
        averageClasses.append(classMatrix/4)
    else:
        averageClasses.append(classMatrix/14)


for i, class1 in enumerate(averageClasses):
    for j, class2 in enumerate(averageClasses):
        print(i, ":", j)
        if np.size(class1, 1) > np.size(class2, 1):
            diff = abs(class2 - class1[:, :np.size(class2, 1)])
        else:
            diff = abs(class1 - class2[:, :np.size(class1, 1)])
        distnaceMatrix[i, j] = np.mean(diff)

for i in distnaceMatrix:
    for j in i:
        print(j),
    print()
print(distnaceMatrix)

# for folder in os.listdir(datasetFolderPath):
#     print(folder)
#     for folder2 in os.listdir(datasetFolderPath):
#         print(folder2)
#         for filename in os.listdir(datasetFolderPath + folder):
#             y, sr = librosa.load(datasetFolderPath + folder + '/' + filename)
#             for filename2 in os.listdir(datasetFolderPath + folder2):
#                 y2, sr2 = librosa.load(datasetFolderPath + folder2 + '/' + filename2)
#                 mfcc = fe.get_mel(y, sr)
#                 mfcc2 = fe.get_mel(y2, sr2)
#                 if np.size(mfcc, 1) > np.size(mfcc2, 1):
#                     diff = abs(mfcc2 - mfcc[:,:np.size(mfcc2, 1)])
#                 else:
#                     diff = abs(mfcc - mfcc2[:,:np.size(mfcc, 1)])
#                 distnaceMatrix[labels.get(folder), labels.get(folder2)] += np.mean(diff)
#                 averageCounter[labels.get(folder), labels.get(folder2)] += 1
#                 break
#             break
# print(distnaceMatrix)
# print(averageCounter)
