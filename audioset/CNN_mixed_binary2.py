from __future__ import print_function
import numpy as np
import csv
import pickle
import operator
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import vggish_params
import vggish_slim
import read_in_dataset
from tensorflow.python.ops import nn_ops
from sklearn.neural_network import MLPClassifier



flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_epochs', 1,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If Frue, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_CLASSES = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
_NUM_CLASSES = 2
_BATCH_SIZE = 50
_RETRAIN = True
_RECOMPUTE_BINARY_PASS = True
_RETEST = True
_REEVALUATE = True
_REGRAPH = True
_THRESHOLD_INCREMENTATION = 0.01


def _get_examples_batch(features, labels, index):
    if index + _BATCH_SIZE < len(features):
        batch = (features[index:index+_BATCH_SIZE], labels[index:index+_BATCH_SIZE])
    else:
        batch = (features[index:], labels[index:])
    return batch


def _test_model_mixed(testPredictions, tLabels):
    if all(testPredictions == tLabels):
        return 100.0
    else:
        for i, t in enumerate(testPredictions):
            if t != 0 and tLabels[i] != 0:
                return 50.0
        return 0.0


def _test_model(predictions, vLabels, sess, classes, classIndex):
    #with open("/cs/scratch/pmh20/predictions.csv", "w+") as my_csv:
    #    csvWriter = csv.writer(my_csv, delimiter=',')
    #    csvWriter.writerows(predictions)

    maxes = np.argmax(predictions, axis=1)
    classification = predictions.astype(int)

    for i, _ in enumerate(classification):
        for j, _ in enumerate(classification[i]):
            classification[i][j] = 0
            classification[i][maxes[i]] = 1

    comparison, correctLabels, predictedLabels = [], [], []
    for i, l in enumerate(vLabels):
        for j, n in enumerate(l):
            if n == 1:
                correctLabels.append(j)
            if classification[i][j] == 1:
                predictedLabels.append(j)
        if all(vLabels[i] == classification[i]):
            comparison.append(1)
        else:
            comparison.append(0)
    conf_matrix = sess.run(tf.confusion_matrix(correctLabels, predictedLabels, num_classes=_NUM_CLASSES))
    print(conf_matrix)
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    plt.subplot(4, 5, classIndex + 1)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    if classIndex + 1 == len(_CLASSES)/4 or classIndex + 1 == len(_CLASSES)/2 or classIndex + 1 == len(_CLASSES)/4*3 or classIndex + 1 == len(_CLASSES):
        plt.colorbar()
    tick_marks = np.arange(_NUM_CLASSES)
    plt.xticks(tick_marks, classes, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6, rotation=90, va='center')
    normalise = True
    if normalise:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalise else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center", verticalalignment="center", fontsize=12,
                 color="white" if conf_matrix[i, j] > thresh else "black")
    if classIndex + 1 == 1 or classIndex + 1 == len(_CLASSES)/2 + 1 or classIndex + 1 == len(_CLASSES)/4 + 1 or classIndex + 1 == len(_CLASSES)/4*3 + 1:
        plt.ylabel('True label', fontsize=15)
    if classIndex + 1 > len(_CLASSES)/4*3:
        plt.xlabel('Predicted label', fontsize=12)
    print("\nActual Labels:   \n", correctLabels, "\n")
    print("\nPredicted Labels:\n", predictedLabels, "\n")
    print("\nComparison:\n", comparison, "\n")
    accuracy = (sum(comparison) / len(comparison)) * 100.0
    print("\nValidation Accuracy: ", accuracy, "%\n\n")
    if classIndex + 1 == 3:
        plt.title('Confusion Matrices', fontsize=30)
    if classIndex + 1 == len(_CLASSES):
        plt.show()


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)
        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope('mymodel'):
            # Add a fully connected layer with 100 units.
            num_units = 100
            fc = slim.fully_connected(embeddings, num_units, activation_fn=nn_ops.relu)
            fc2 = slim.fully_connected(fc, 60, activation_fn=nn_ops.relu)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            fc3 = slim.fully_connected(
                fc2, _NUM_CLASSES, activation_fn=None, scope='logits')
            softM = tf.nn.softmax(fc3)
            predictionTest = softM
            # Add training ops.
            with tf.variable_scope('train'):
                global_step = tf.Variable(
                    0, name='global_step', trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.GLOBAL_STEP])

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(
                    tf.float32, shape=(None, _NUM_CLASSES), name='labels')

                # Cross-entropy label loss.
                xent = tf.nn.softmax_cross_entropy_with_logits(
                    logits=softM, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)
                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=vggish_params.LEARNING_RATE,
                    epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

        if _RETRAIN:
            saver = tf.train.Saver()
            # Initialize all variables in the model, and then load the pre-trained VGGish checkpoint.
            sess.run(tf.global_variables_initializer())
            vggish_slim.load_vggish_slim_checkpoint(sess, "/cs/home/pmh20/workspace_linux/Project/CNN/vggish/models-master/research/audioset/vggish_model.ckpt")

            # Locate all the tensors and ops
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)

            labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
            global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
            loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
            tf.summary.scalar('loss', loss_tensor)
            train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

            with open("/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/sameCluster/Cluster3/test_Mix-5.txt", "rb") as fp:
            #with open("/cs/tmp/pmh20/a/test_TestDatasetWild.txt", "rb") as fp:
                (features, labels) = pickle.load(fp)
            
            classes = ['Car', 'Chair', 'ClearThroat', 'CoffeeMachine', 'Conversation', 'Coughing', 'Dishwasher', 'DoorKnock', 'DoorSlam', 'Drawer', 'FallingObject', 'FootSteps', 'Keyboard', 'Laughing', 'MilkSteamer', 'PhoneRinging', 'Photocopier', 'Sink', 'Sneezing', 'Stiring']
           
            if _RECOMPUTE_BINARY_PASS:
                results, correctLabels = [], []
                for classIndex, singleClass in enumerate(_CLASSES):
                    print('----', singleClass, '----')
                    saver.restore(sess, tf.train.latest_checkpoint('/cs/tmp/pmh20/ExtraWork/BinaryModels/' + singleClass + '/'))
                    singleClassifierResults = []
                    for fileIndex, singleFile in enumerate(features):
                        if len(singleFile) > 0:
                            testPredictions = sess.run(predictionTest, feed_dict={features_tensor: singleFile})
                            certainty = np.mean(testPredictions, axis=0)[0]
                            singleClassifierResults.append(certainty)
                            print(classIndex, ':', singleClass, ':', fileIndex)
                            if classIndex == 0:
                                correctLabels.append(labels[fileIndex])
                    results.append(singleClassifierResults)
                results = list(map(list, zip(*results)))
                #print(results)
                with open("/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/sameCluster/Cluster3/outputs+labels_test_m5.txt", "wb") as fp:
     #           with open("/cs/tmp/pmh20/a/outputs+labels_test.txt", "wb") as fp:
                    pickle.dump((results, correctLabels), fp)

            #Training sencond classifier, which uses the binary classifier's results as inputs, and trains on both singles and mixes.
            with open("/cs/tmp/pmh20/ExtraWork/outputs+labels_train.txt", "rb") as fp:
                (x_train, y_train) = pickle.load(fp)
            with open("/cs/tmp/pmh20/ExtraWork/AugmentedDataset/TestSet/sameCluster/Cluster3/outputs+labels_test_m5.txt", "rb") as fp:
        #    with open("/cs/tmp/pmh20/a/outputs+labels_test.txt", "rb") as fp:            
                (x_val, y_val) = pickle.load(fp)
            #with open("/cs/tmp/pmh20/ExtraWork/outputs+labels_test.txt", "rb") as fp:
            #    (x_test, y_test) = pickle.load(fp)

            second_classifier = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(20, 17, 12, 17, 20, ), learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.7,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0, verbose=False,
       warm_start=False)
            second_classifier.fit(x_train, y_train)
            y_pred = second_classifier.predict_proba(x_val)
            y_pred = np.array(y_pred)
            y_val = np.array(y_val)
            # test_accuracy = 100 * second_classifier.score(x_test, y_test)
            # print('Accuracy = ', test_accuracy)
            with open("/cs/tmp/pmh20/ExtraWork/predictions+labels_test.txt", "wb") as fp:
                pickle.dump((y_pred, y_val), fp)


    if _REEVALUATE:
        with open("/cs/tmp/pmh20/ExtraWork/predictions+labels_test.txt", "rb") as fp:
            (results, correctLabels) = pickle.load(fp)

        threshold, precisionList, recallList, accuracyList, f1ScoreList, thresholdList = 0.0, [], [], [], [], []
        while threshold <= 0.99:
            truePositives, falsePositives, trueNegatives, falseNegatives = 0, 0, 0, 0
            for i, lab in enumerate(correctLabels):
                for j, val in enumerate(results[i]):
                    if val >= threshold and lab[j] == 1:
                        truePositives += 1
                    elif val >= threshold and lab[j] == 0:
                        falsePositives += 1
                    elif val < threshold and lab[j] == 0:
                        trueNegatives += 1
                    elif val < threshold and lab[j] == 1:
                        falseNegatives += 1
            precision = truePositives / (truePositives + falsePositives)
            recall = truePositives / (truePositives + falseNegatives)
            accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
            f1Score = (2 * precision * recall) / (precision + recall)
            precisionList.append(precision)
            recallList.append(recall)
            accuracyList.append(accuracy)
            f1ScoreList.append(f1Score)
            thresholdList.append(threshold)
            threshold += _THRESHOLD_INCREMENTATION
            threshold = float("{0:.2f}".format(threshold))

        index, value = max(enumerate(f1ScoreList), key=operator.itemgetter(1))
        maxF1 = thresholdList[index]
        optimalThreshold = [maxF1, maxF1]
        yThres = [0, 1]
        #index = 0
        print('Precision:', precisionList[index])
        print('Recall:', recallList[index])
        print('Accuracy:', accuracyList[index])
        print('F1 Score:', f1ScoreList[index])
        print('Optimal Threshold:', maxF1)

        if _REGRAPH:
            plt.subplot(2, 1, 1)
            plt.title('Precision / Recall over increasing Thresholds', fontsize=16)
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel('Precision / Recall Score', fontsize=12)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.plot(thresholdList, precisionList, lw=2, c='b', label='Precision')
            plt.plot(thresholdList, recallList, lw=2, c='orange', label='Recall')
            plt.plot(optimalThreshold, yThres, 'k--', lw=2, label='Optimal Threshold (' + str(maxF1) + ')')
            plt.legend(loc='lower center')
            plt.grid()
            plt.subplot(2, 1, 2)
            plt.title('F1-Score / Accuracy over increasing Thresholds', fontsize=16)
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel('Accuracy / F1 Score', fontsize=12)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.plot(thresholdList, accuracyList, lw=2, c='limegreen', label='Accuracy')
            plt.plot(thresholdList, f1ScoreList, lw=2, c='r', label='F1-Score')
            plt.plot(optimalThreshold, yThres, 'k--', lw=2, label='Optimal Threshold (' + str(maxF1) + ')')
            plt.legend(loc='lower center')
            plt.grid()
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    tf.app.run()
