import os
import vggish_input
import soundfile as sf	 
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
from scipy import stats


datasetFolderPath = "/home/.scratch_space/pmh20/ExtraWork/OriginalDataset/"
features = []
#
#for m, folder in enumerate(os.listdir(datasetFolderPath)):
#    print(m)
#    d, sr= [], []
#    for filename in os.listdir(datasetFolderPath + folder):
#        data, samplerate = sf.read("/home/.scratch_space/pmh20/ExtraWork/OriginalDataset/" + folder + "/" + filename)
#        d.append(data)
#        sr.append(samplerate)        
#    import random
#    r = random.sample(range(1, 13), 4)
#    j, k, l = 0, 0, 0
#    for i, s in enumerate(d):
#        if (i == r[0] or i == r[1]) and folder != "MilkSteamer":
#            sf.write("/home/.scratch_space/pmh20/ExtraWork/TestSet/" + folder + "/" + folder + "_" + str(j) + ".wav", s, sr[i], subtype='PCM_16', format="wav")
#            j += 1        
#        elif (i == r[2] or i == r[3]) and folder != "MilkSteamer":
#            sf.write("/home/.scratch_space/pmh20/ExtraWork/ValidationSet/" + folder + "/" + folder + "_" + str(k) + ".wav", s, sr[i], subtype='PCM_16', format="wav")
#            k += 1
#        else:
#            sf.write("/home/.scratch_space/pmh20/ExtraWork/TrainingSet/" + folder + "/" + folder + "_" + str(l) + ".wav", s, sr[i], subtype='PCM_16', format="wav")
#            l += 1
#for i, folder in enumerate(os.listdir(datasetFolderPath)):
#    print(i)
#    for filename in os.listdir(datasetFolderPath + folder):
#        feat = vggish_input.wavfile_to_examples(datasetFolderPath + folder + "/" + filename)
#        for ft in feat:
#            features.append(ft)
#
#with open('/home/.scratch_space/pmh20/ExtraWork/KM_features.txt', 'wb') as handle:
#    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/.scratch_space/pmh20/ExtraWork/KM_features.txt', 'rb') as handle:
    features = pickle.load(handle)

x = []
for f in features:
    x.append(f.flatten())
X = np.array(x)

# k means determine k
#distortions = []
#K = range(1, 10)
#for k in K:
#    print(k)
k = 4
kmeanModel = KMeans(n_clusters=k).fit(X)
kmeanModel.fit(X)
#    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

#print(distortions)

datasetPath = '/home/.scratch_space/pmh20/ExtraWork/'
datasetName = 'AugmentedDataset/TestSet/'
features = []
for i, folder in enumerate(os.listdir(datasetPath + datasetName)):
    print(i)
    for filename in os.listdir(datasetPath + datasetName + folder):
        data, samplerate = sf.read(datasetPath + datasetName + folder + "/" + filename)
        feat = vggish_input.wavfile_to_examples(datasetPath + datasetName + folder + "/" + filename)
        for ft in feat:
            features.append(ft)
        x = []
        for f in features:
            x.append(f.flatten())
        test_X = np.array(x)
        predictions = kmeanModel.predict(test_X)
        mode = stats.mode(predictions)[0]
        cluster = mode[0] + 1
        print(i, " -  Cluster:", cluster)        
        sf.write(datasetPath + "Clusters/Cluster" + str(cluster) + "/" + folder + "/" + filename, data, samplerate, subtype='PCM_16', format="wav")









