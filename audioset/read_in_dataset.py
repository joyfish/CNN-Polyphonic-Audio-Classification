import pickle
import zipfile
import numpy as np
import sklearn.model_selection as sk
import vggish_input
import os
import blind_source_separation
from os.path import basename, splitext


def _repeat_blind_separation():
    ftsUnmixed1, ftsUnmixed2, ftsUnmixed3, ftsUnmixed4 = [], [], [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/MixedEventsDataset.zip", "r") as dataset:
        clas, counter, n = "", 0, 0
        for f in dataset.namelist():
            if "." not in f and len(f[18:-1]) > 0:
                clas = f[18:-1]
                counter += 1
            if "." in f:
                print(counter, ": ", f)
                name = splitext(basename(f))[0]
                fullPath = "/cs/scratch/pmh20/Dataset/MixedEventsDataset/" + str(clas) + "/" + name + ".wav"
                unmixedEvents = blind_source_separation.fast_ICA(fullPath, 2)
                unmixedEvents2 = blind_source_separation.high_low_pass_filters(fullPath)
                ftsUnmixed1.append(unmixedEvents[0])
                ftsUnmixed2.append(unmixedEvents[1])
                ftsUnmixed3.append(unmixedEvents2[0])
                ftsUnmixed4.append(unmixedEvents2[1])
    x_test_split1 = np.array(ftsUnmixed1)
    x_test_split2 = np.array(ftsUnmixed2)
    x_test_split3 = np.array(ftsUnmixed3)
    x_test_split4 = np.array(ftsUnmixed4)
    with open("/cs/scratch/pmh20/Dataset/features_split1.txt", "wb") as fp:
        pickle.dump(x_test_split1, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split2.txt", "wb") as fp:
        pickle.dump(x_test_split2, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split3.txt", "wb") as fp:
        pickle.dump(x_test_split3, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split4.txt", "wb") as fp:
        pickle.dump(x_test_split4, fp)
    return (x_test_split1, x_test_split2, x_test_split3, x_test_split4)


def _get_train_test_split_saved_mixed(repeat=False):
    labelList = []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/OriginalDataset.zip", "r") as dataset:
        for f in dataset.namelist():
            if "." not in f and len(f[16:-1]) > 0:
                labelList.append(f[16:-1])
    labelList = sorted(labelList)
    with open("/cs/scratch/pmh20/Dataset/features.txt", "rb") as fp:
        features = pickle.load(fp)
    with open("/cs/scratch/pmh20/Dataset/labels.txt", "rb") as fp:
        labels = pickle.load(fp)
    with open("/cs/scratch/pmh20/Dataset/features_mixed_test.txt", "rb") as fp:
        x_test = pickle.load(fp)
    with open("/cs/scratch/pmh20/Dataset/labels_mixed_test.txt", "rb") as fp:
        y_test = pickle.load(fp)
    if repeat:
        (x_test_split1, x_test_split2, x_test_split3, x_test_split4) = _repeat_blind_separation()
    else:
        with open("/cs/scratch/pmh20/Dataset/features_split1.txt", "rb") as fp:
            x_test_split1 = pickle.load(fp)
        with open("/cs/scratch/pmh20/Dataset/features_split2.txt", "rb") as fp:
            x_test_split2 = pickle.load(fp)
        with open("/cs/scratch/pmh20/Dataset/features_split3.txt", "rb") as fp:
            x_test_split3 = pickle.load(fp)
        with open("/cs/scratch/pmh20/Dataset/features_split4.txt", "rb") as fp:
            x_test_split4 = pickle.load(fp)

    x_train, x_validate, y_train, y_validate = sk.train_test_split(features, labels, test_size=0.1)
    split = [labelList, (x_train, y_train), (x_validate, y_validate), (x_test, y_test), (x_test_split1, x_test_split2), (x_test_split3, x_test_split4)]
    print(features.shape)
    return split


def _get_train_test_split_new_mixed(maxSeparations):
    labelList = []
    encodedLabels = {}
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/OriginalDataset.zip", "r") as dataset:
        clas = ""
        for f in dataset.namelist():
            if "." not in f and len(f[16:-1]) > 0:
                clas = f[16:-1]
                labelList.append(clas)
                labelList = sorted(labelList)
    for i, l in enumerate(labelList):
        encoding = [0 for _ in labelList]
        encoding[i] = 1
        encodedLabels[labelList[i]] = encoding

    ft, l = [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/OriginalDataset.zip", "r") as dataset:
        clas, counter, n = "", 0, 0
        for f in dataset.namelist():
            if "." not in f and len(f[16:-1]) > 0:
                clas = f[16:-1]
                counter += 1
                labelList.append(clas)
                labelList = sorted(labelList)
            if "." in f:
                print(counter, ": ", f)
                fts = (vggish_input.wavfile_to_examples("/cs/scratch/pmh20/Dataset/OriginalDataset/" + clas + "/" + splitext(basename(f))[0] + ".wav"))
                for feat in fts:
                    ft.append(feat)
                    l.append(encodedLabels[clas])
    features = np.array(ft)
    labels = np.array(l)

    ftsOriginal, lByFile, ftsUnmixed1, ftsUnmixed2, ftsUnmixed3, ftsUnmixed4 = [], [], [], [], [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/EasyMixes.zip", "r") as dataset:
        clas, counter, n = "", 0, 0
        for f in dataset.namelist():
            if "." not in f and len(f[9:-1]) > 0:
                clas = f[9:-1]
                counter += 1
            if "." in f:
                print(counter, ": ", f)
                name = splitext(basename(f))[0]
                fullPath = "/cs/scratch/pmh20/Dataset/EasyMixes/" + str(clas) + "/" + name + ".wav"
                ftsOriginal.append(vggish_input.wavfile_to_examples(fullPath))
                classes = clas[1:].split("_+_")
                sourceClasses = []
                for c in classes:
                    sourceClasses.append(encodedLabels[c])
                combinedClasses = [sum(x) for x in zip(*sourceClasses)]
                lByFile.append(combinedClasses)
                for index in range(maxSeparations - 1):
                    i = index + 1
                    numberOfEvents = i + 1
                    unmixedEvents = blind_source_separation.fast_ICA(fullPath, numberOfEvents)
                    unmixedEvents2 = blind_source_separation.high_low_pass_filters(fullPath)
                ftsUnmixed1.append(unmixedEvents[0])
                ftsUnmixed2.append(unmixedEvents[1])
                ftsUnmixed3.append(unmixedEvents2[0])
                ftsUnmixed4.append(unmixedEvents2[1])
    x_test = np.array(ftsOriginal)
    x_test_split1 = np.array(ftsUnmixed1)
    x_test_split2 = np.array(ftsUnmixed2)
    x_test_split3 = np.array(ftsUnmixed3)
    x_test_split4 = np.array(ftsUnmixed4)
    y_test = np.array(lByFile)

    with open("/cs/scratch/pmh20/Dataset/features.txt", "wb") as fp:
        pickle.dump(features, fp)
    with open("/cs/scratch/pmh20/Dataset/labels.txt", "wb") as fp:
        pickle.dump(labels, fp)
    with open("/cs/scratch/pmh20/Dataset/features_mixed_test.txt", "wb") as fp:
        pickle.dump(x_test, fp)
    with open("/cs/scratch/pmh20/Dataset/labels_mixed_test.txt", "wb") as fp:
        pickle.dump(y_test, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split1.txt", "wb") as fp:
        pickle.dump(x_test_split1, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split2.txt", "wb") as fp:
        pickle.dump(x_test_split2, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split3.txt", "wb") as fp:
        pickle.dump(x_test_split3, fp)
    with open("/cs/scratch/pmh20/Dataset/features_split4.txt", "wb") as fp:
        pickle.dump(x_test_split4, fp)

    x_train, x_validate, y_train, y_validate = sk.train_test_split(features, labels, test_size=0.1)
    split = [labelList, (x_train, y_train), (x_validate, y_validate), (x_test, y_test), (x_test_split1, x_test_split2), (x_test_split3, x_test_split4)]
    print(features.shape)
    return split


def _get_train_test_split_saved():
    labelList = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
    with open("/cs/scratch/pmh20/Dataset/SingleEventSplit.txt", "rb") as fp:
        [(x_train, y_train), (x_validate, y_validate), (x_test, y_test)] = pickle.load(fp)
    x_train2, y_train2, x_validate2, y_validate2, x_test2, y_test2 = [], [], [], [], [], []
    for i, file in enumerate(x_train):
        for sample in file:
            x_train2.append(sample)
            y_train2.append(y_train[i])
    for i, file in enumerate(x_validate):
        for sample in file:
            x_validate2.append(sample)
            y_validate2.append(y_validate[i])
    for i, file in enumerate(x_test):
        for sample in file:
            x_test2.append(sample)
            y_test2.append(y_test[i])
    split = [(x_train2, y_train2), (x_validate2, y_validate2), (x_test2, y_test2)]

    with open(os.getcwd() + "/TestingModel/TestDataset-Preprocessed.txt", "wb") as fp:
        pickle.dump((x_test2, y_test2), fp)

    return split


def _get_train_test_split_new():
    labelList = []
    encodedLabels = {}
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/TestDatasetWild.zip", "r") as dataset:
        clas = ""
        for f in dataset.namelist():
            if "." not in f and len(f[16:-1]) > 0:
                clas = f[16:-1]
                labelList.append(clas)
                labelList = sorted(labelList)
    for i, l in enumerate(labelList):
        encoding = [0 for _ in labelList]
        encoding[i] = 1
        encodedLabels[labelList[i]] = encoding

    ft, l = [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/TestDatasetWild.zip", "r") as dataset:
        clas, counter = "", 0
        for f in dataset.namelist():
            if "." not in f and len(f[16:-1]) > 0:
                clas = f[16:-1]
                counter += 1
            if "." in f:
                try:
                    fts = (vggish_input.wavfile_to_examples("/cs/scratch/pmh20/Dataset/TestDatasetWild/" + clas + "/" + splitext(basename(f))[0] + ".wav"))
                    for feat in fts:
                        ft.append(feat)
                        l.append(encodedLabels[clas])
                    print(counter, ": ", f)
                    print(clas, encodedLabels[clas])
                except: 1
    features = np.array(ft)
    labels = np.array(l)

    x_train, x_rest, y_train, y_rest = sk.train_test_split(features, labels, test_size=0.3)
    x_validate, x_test, y_validate, y_test = sk.train_test_split(x_rest, y_rest, test_size=0.5)
    split = [(x_train, y_train), (x_validate, y_validate), (x_test, y_test)]

    with open("/cs/scratch/pmh20/Dataset/TestDatasetWild.txt", "wb") as fp:
        pickle.dump((features, labels), fp)

    with open("/cs/scratch/pmh20/Dataset/SingleEventSplit.txt", "rb") as fp:
        [(x_train, y_train), (x_validate, y_validate), (x_test, y_test)] = pickle.load(fp)
    x_train2, y_train2, x_validate2, y_validate2, x_test2, y_test2 = [], [], [], [], [], []
    for i, file in enumerate(x_train):
        for sample in file:
            x_train2.append(sample)
            y_train2.append(y_train[i])
    for i, file in enumerate(x_validate):
        for sample in file:
            x_validate2.append(sample)
            y_validate2.append(y_validate[i])
    for i, file in enumerate(x_test):
        for sample in file:
            x_train2.append(sample)
            y_train2.append(y_test[i])
    split = [(x_train2, y_train2), (x_validate2, y_validate2), (features, labels)]

    print(features.shape)
    return split



def _get_train_test_split_saved_combos():
    labelList = []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes.zip", "r") as dataset:
        for f in dataset.namelist():
            if "." not in f and len(f[:-1]) > 0:
                labelList.append(f[:-1])
    labelList = sorted(labelList)
    with open("/cs/scratch/pmh20/Dataset/featuresInclCombos.txt", "rb") as fp:
        features = pickle.load(fp)
    with open("/cs/scratch/pmh20/Dataset/labelsCombos.txt", "rb") as fp:
        labels = pickle.load(fp)
    x_train, x_validate, y_train, y_validate = sk.train_test_split(features, labels, test_size=0.3)
    split = [labelList, (x_train, y_train), (x_validate, y_validate)]
    print(features.shape)
    return split


def _get_train_test_split_new_combos():
    labelList = []
    encodedLabels = {}
    # import shutil
    # shutil.make_archive('/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes', 'zip', '/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes')
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes.zip", "r") as dataset:
        clas = ""
        for f in dataset.namelist():
            if "." not in f and len(f[:-1]) > 0 and f != 'Conversation_+_Dishwasher/Conversation_13_AND_Dishwasher_20.wav':
                clas = f[:-1]
                labelList.append(clas)
                labelList = sorted(labelList)
    for i, l in enumerate(labelList):
        encoding = [0 for _ in labelList]
        encoding[i] = 1
        encodedLabels[labelList[i]] = encoding

    ft, l = [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes.zip", "r") as dataset:
        clas, oldClas, counter, n = "", "", 1, 0
        for f in dataset.namelist():
            if "." in f and f != 'Conversation_+_Dishwasher/Conversation_13_AND_Dishwasher_20.wav':
                clas = f.split('/')[0]
                if clas != oldClas and len(oldClas) > 0:
                    counter += 1
                oldClas = clas
                print(counter, ": ", f)
                fts = (vggish_input.wavfile_to_examples("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes/" + f))
                for feat in fts:
                    ft.append(feat)
                    l.append(encodedLabels[clas])
    features = np.array(ft)
    labels = np.array(l)
    with open("/cs/scratch/pmh20/Dataset/featuresInclCombos.txt", "wb") as fp:
        pickle.dump(features, fp)
    with open("/cs/scratch/pmh20/Dataset/labelsCombos.txt", "wb") as fp:
        pickle.dump(labels, fp)

    x_train, x_validate, y_train, y_validate = sk.train_test_split(features, labels, test_size=0.15)
    split = [labelList, (x_train, y_train), (x_validate, y_validate)]
    print(features.shape)
    return split


def _get_train_test_split_saved_binary(c, nomix=False):
    labelList = [c, 'Not ' + c]
    encodingList = []
    for i in range(2):
        encoding = [0 for _ in range(2)]
        encoding[i] = 1
        encodingList.append(encoding)

        if nomix:
            path = "/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier1/"
        else:
            path = "/cs/scratch/pmh20/Dataset/BinaryClassifier1/"

    with open(path + "featuresBinaryTrain" + c + ".txt", "rb") as fp:
        x_train = pickle.load(fp)
    with open(path + "labelsBinaryTrain" + c + ".txt", "rb") as fp:
        y_train = pickle.load(fp)
    with open(path + "featuresBinaryVal" + c + ".txt", "rb") as fp:
        x_validate = pickle.load(fp)
    with open(path + "labelsBinaryVal" + c + ".txt", "rb") as fp:
        y_validate = pickle.load(fp)
    split = [labelList, (x_train, y_train), (x_validate, y_validate)]
    return split


def _get_train_test_split_new_binary(singleClass, nomix=False):
    classes = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
    labelList = [singleClass, 'Not ' + singleClass]
    encodingList = []
    for i in range(2):
        encoding = [0 for _ in range(2)]
        encoding[i] = 1
        encodingList.append(encoding)

    if nomix:
        path = "/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier1/"
    else:
        path = "/cs/scratch/pmh20/Dataset/BinaryClassifier1/"

    with open(path + "filePathList.txt", "rb") as fp:
        features = pickle.load(fp)
    x, x_names_test = sk.train_test_split(features, test_size=0.15)
    x_names_train, x_names_validate = sk.train_test_split(x, test_size=0.15)

    for c in classes:
        ft, l, counter, n, n2 = [], [], 1, 0, 0
        for name in x_names_train:
            clas = name[1:].split('/')[5]
            #filePaths.append("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes/" + f)
            if nomix:
                if c == clas or n2 < n:
                    print(counter, ": ", name)
                    fts = (vggish_input.wavfile_to_examples(name))
                    for feat in fts:
                        if c == clas:
                            l.append(encodingList[0])
                            n += 1
                            counter += 1
                            ft.append(feat)
                        else:
                            if n2 < n:
                                l.append(encodingList[1])
                                ft.append(feat)
                                n2 += 1
                                counter += 1
            else:
                if c in clas or n2 < n:
                    print(counter, ": ", name)
                    fts = (vggish_input.wavfile_to_examples(name))
                    for feat in fts:
                        if c in clas:
                            l.append(encodingList[0])
                            n += 1
                            counter += 1
                            ft.append(feat)
                        else:
                            if n2 < n:
                                l.append(encodingList[1])
                                ft.append(feat)
                                n2 += 1
                                counter += 1
        print(c, ':', n, '  Not', c, ':', n2)
        x_train = np.array(ft)
        y_train = np.array(l)

        ft, l, counter, n, n2 = [], [], 1, 0, 0
        for name in x_names_validate:
            clas = name[1:].split('/')[5]
            if nomix:
                if c == clas or n2 < n:
                    print(counter, ": ", name)
                    fts = (vggish_input.wavfile_to_examples(name))
                    for feat in fts:
                        if c == clas:
                            l.append(encodingList[0])
                            n += 1
                            counter += 1
                            ft.append(feat)
                        else:
                            if n2 < n:
                                l.append(encodingList[1])
                                ft.append(feat)
                                n2 += 1
                                counter += 1
            else:
                if c in clas or n2 < n:
                    if c == clas or n2 < n:
                        print(counter, ": ", name)
                        fts = (vggish_input.wavfile_to_examples(name))
                        for feat in fts:
                            if c in clas:
                                l.append(encodingList[0])
                                n += 1
                                counter += 1
                                ft.append(feat)
                            else:
                                if n2 < n:
                                    l.append(encodingList[1])
                                    ft.append(feat)
                                    n2 += 1
                                    counter += 1

        print(c, ':', n, '  Not', c, ':', n2)
        x_validate = np.array(ft)
        y_validate = np.array(l)
        try:
            with open(path + "featuresBinaryTrain" + c + ".txt", "wb") as fp:
                pickle.dump(x_train, fp)
            with open(path + "labelsBinaryTrain" + c + ".txt", "wb") as fp:
                pickle.dump(y_train, fp)
            with open(path + "featuresBinaryVal" + c + ".txt", "wb") as fp:
                pickle.dump(x_validate, fp)
            with open(path + "labelsBinaryVal" + c + ".txt", "wb") as fp:
                pickle.dump(y_validate, fp)
        except: print("NOT ENOUGH SPACE")

    x_test, y_test, encodedLabels = [], [], {}

    for i, l in enumerate(classes):
        encoding = [0 for _ in classes]
        encoding[i] = 1
        encodedLabels[classes[i]] = encoding

    for name in x_names_test:
        clas = name[1:].split('/')[5]
        fts = vggish_input.wavfile_to_examples(name)
        x_test.append(fts)
        cls = clas.split("_+_")
        sourceClasses = []
        for cl in cls:
            sourceClasses.append(encodedLabels[cl])
        combinedClasses = [sum(x) for x in zip(*sourceClasses)]
        y_test.append(combinedClasses)
        print(clas, ' ', combinedClasses)

    try:
        with open(path + "featuresBinaryTest.txt", "wb") as fp:
            pickle.dump(x_test, fp)
        with open(path + "labelsBinaryTest.txt", "wb") as fp:
            pickle.dump(y_test, fp)
    except: print("NOT ENOUGH SPACE")

    split = [labelList, (x_train, y_train), (x_validate, y_validate), (x_test, y_test)]
    return split


def _get_train_test_split_saved_binary2_classifier2():
    labelList = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
    with open("/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier2/allFeatures.txt", "rb") as fp:
        features = pickle.load(fp)
    with open("/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier2/allLabels.txt", "rb") as fp:
        labels = pickle.load(fp)
    x_train, x_rest, y_train, y_rest = sk.train_test_split(features, labels, test_size=0.3)
    x_validate, x_test, y_validate, y_test = sk.train_test_split(x_rest, y_rest, test_size=0.5)
    split = [labelList, (x_train, y_train), (x_validate, y_validate), (x_test, y_test)]
    print(features.shape)
    return split


def _get_train_test_split_new_binary2_classifier2():
    classes = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
    encodedLabels, labelList = {}, []
    for i in range(len(classes)):
        encoding = [0 for _ in classes]
        encoding[i] = 1
        encodedLabels[classes[i]] = encoding

    ftByFile, l = [], []
    with zipfile.ZipFile("/cs/scratch/pmh20/Dataset/LargeDatasetWithMixes.zip", "r") as dataset:
        clas, oldClas, counter, n = "", "", 1, 0
        for f in dataset.namelist():
            if "." in f:
                clas = f.split('/')[1]
                if clas != oldClas and len(oldClas) > 0:
                    counter += 1
                oldClas = clas
                if '_+_' in clas:
                    mix_classes = clas.split('_+_') if '_+_' in clas else [clas]
                print(counter, ": ", f)
                fts = (vggish_input.wavfile_to_examples("/cs/scratch/pmh20/Dataset/" + f))
                ft = []
                label = [0] * len(classes)
                for feat in fts:
                    ft.append(feat)
                mix_labels = [encodedLabels[cl] for cl in mix_classes]
                for ml in mix_labels:
                    label = [label[i] + ml[i] for i in range(len(ml))]
                l.append(label)
                ftByFile.append(ft)
    features = np.array(ftByFile)
    labels = np.array(l)
    with open("/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier2/allFeatures.txt", "wb") as fp:
        pickle.dump(features, fp)
    with open("/cs/scratch/pmh20/Dataset/BinaryClassifier2/Classifier2/allLabels.txt", "wb") as fp:
        pickle.dump(labels, fp)

    x_train, x_test, y_train, y_test = sk.train_test_split(features, labels, test_size=0.3)
    split = [labelList, (x_train, y_train), (x_test, y_test)]
    print(features.shape)

    return split
