import random
import os
from pydub import AudioSegment


new_path = '/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/diffClusters/Mix-3/'
number_of_mixes = 3
desired_size = 432000000 * number_of_mixes
k = 0
while(True):
    clusters = random.sample([1, 2, 3], number_of_mixes)
    files_to_mix, name = [], ""
    for clust in clusters:
        dataset_folder_path = '/cs/tmp/pmh20/ExtraWork/Clusters/Cluster' + str(clust) + '/'
#        print(len(os.listdir(dataset_folder_path)), cluster)
        folder_selection = random.sample(range(len(os.listdir(dataset_folder_path))), 1)
        for i, folder in enumerate(os.listdir(dataset_folder_path)):
#            print(i, folder)
            if i in folder_selection and ".DS_Store" not in folder:
                file_selection = random.randint(0, len(os.listdir(dataset_folder_path + folder)) - 1)
                for j, filename in enumerate(os.listdir(dataset_folder_path + folder)):
                    if j == file_selection and ".DS_Store" not in filename:
                        files_to_mix.append(AudioSegment.from_file(dataset_folder_path + folder + "/" + filename))
                        if len(name) == 0:
                            name = folder
                        else:
                            name += "_+_" + folder
    for i, f in enumerate(files_to_mix):
        if i == 0:
            file1 = f
        else:
            file2 = f
            loopNumber = int(max(len(file1), len(file2)) / min(len(file1), len(file2)))
            if len(file1) < len(file2):
                file1 = file1.__mul__(loopNumber)
            elif len(file2) < len(file1):
                file2 = file2.__mul__(loopNumber)
            combined = file1.overlay(file2)
            file1 = combined
    file1.export(new_path + name + "_" + str(k) + ".wav", format="wav")
    size = sum(os.path.getsize(new_path + "/" + f) for f in os.listdir(new_path) if os.path.isfile(new_path + "/" + f))
    print(size, ":", name)
    if size >= desired_size:
        break
    k += 1



















