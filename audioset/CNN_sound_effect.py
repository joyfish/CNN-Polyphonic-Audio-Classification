from __future__ import print_function
import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import vggish_params
import vggish_slim
import itertools
import read_in_dataset


flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_epochs', 30,
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
_NUM_CLASSES = 20
_BATCH_SIZE = 650
_RETRAIN = False


# This function takes the features and labels, and based
# on the current loop index, will return a certain batch.
def _get_examples_batch(features, labels, index):
    if index + _BATCH_SIZE < len(features):
        batch = (features[index:index+_BATCH_SIZE], labels[index:index+_BATCH_SIZE])
    else:
        batch = (features[index:], labels[index:])
    return batch


# This function is used for testing the output results against
# the ground truth (correct) labels. It takes predictions and
# test labels as arguments and compares the arrays. It also has
# and optional argument that allows the confusion matrix to be shown.
def _test_model(testPredictions, tLabels, sess, confM=False):

    maxes = np.argmax(testPredictions, axis=1)
    classification = testPredictions.astype(int)

    # This loop sets an array to the classifications that were made.
    # It is in integers to make it directly comparable to the labels.
    for i, _ in enumerate(classification):
        for j, _ in enumerate(classification[i]):
            classification[i][j] = 0
            classification[i][maxes[i]] = 1

    comparison, correctLabels, predictedLabels = [], [], []

    # Creates two 1-D lists that hold the indices of the predicted and
    # true labels. These are printed later for a visual comparison.
    for i, l in enumerate(tLabels):
        for j, n in enumerate(l):
            if n == 1:
                correctLabels.append(j)
            if classification[i][j] == 1:
                predictedLabels.append(j)

        # Here the output layer is compared
        # to the labels, sample by sample.
        if all(tLabels[i] == classification[i]):
            comparison.append(1)
        else:
            comparison.append(0)

    # If desired, the confusion matrix is
    # calculated here, and displayed to the screen.
    if confM:
        conf_matrix = sess.run(tf.confusion_matrix(correctLabels, predictedLabels, num_classes=_NUM_CLASSES))

        # Normalises the values so the colour mapping still works with unequally weighted classes.
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.subplot(1, 1, 1)  # Previously, more than one subplot was used.
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        plt.colorbar()
        tick_marks = np.arange(_NUM_CLASSES)
        plt.xticks(tick_marks, _CLASSES, fontsize=10, rotation=90)
        plt.yticks(tick_marks, _CLASSES, fontsize=10, va='center')
        formatting = '.2f'
        thresh = conf_matrix.max() / 2.

        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, format(conf_matrix[i, j], formatting), horizontalalignment="center", verticalalignment="center", fontsize=12, color="white" if conf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=12)
        print("\nActual Labels:   \n", correctLabels, "\n")
        print("\nPredicted Labels:\n", predictedLabels, "\n")
        print("\nComparison:\n", comparison, "\n")
        accuracy = (sum(comparison) / len(comparison)) * 100.0
        print("\nTesting Accuracy: ", accuracy, "%\n\n")
        plt.title('Confusion Matrix', fontsize=30)
        plt.tight_layout()
        plt.show()

    accuracy = (sum(comparison) / len(comparison)) * 100.0
    print("\nAccuracy: ", accuracy, "%\n\n")
    return accuracy


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:

        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

        # Define a shallow classification model and
        # associated training ops on top of VGGish.
        with tf.variable_scope('mymodel'):
            # Add fully connected layers.
            num_units = 200
            fc = slim.fully_connected(embeddings, num_units)
            fc2 = slim.fully_connected(fc, int(num_units/2))

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            fc3 = slim.fully_connected(fc2, _NUM_CLASSES, activation_fn=None, scope='logits')

            # Create a Softmax layer to
            # ensure outputs always sum to 1.
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

                # Cross-entropy label loss, specifically one for Softmax.
                xent = tf.nn.softmax_cross_entropy_with_logits(logits=softM, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)

                # The same optimizer (Adam) and hyperparameters are used that were used to train VGGish.
                optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')


        if _RETRAIN == False:
            saver = tf.train.Saver(tf.global_variables())
            with open(os.getcwd() + "/TestingModel/LearningCurveFromTraining.txt", "rb") as fp:
                (learningCurveInd, learningCurveTrn, learningCurveVal) = pickle.load(fp)
            with open(os.getcwd() + "/TestingModel/TestDataset-Preprocessed.txt", "rb") as fp:
                (tFeatures, tLabels) = pickle.load(fp)


        # initialise the saver for saving the model.
        if _RETRAIN:
            saver = tf.train.Saver()

        # Initialize all variables in the model, and
        # then load the pre-trained VGGish checkpoint.
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        tf.summary.scalar('loss', loss_tensor)
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        if _RETRAIN:
            # Uses an additional module that was created for returning the
            # train/val/test split. different function name for each approach.
            trainTestSplit = read_in_dataset._get_train_test_split_saved()
            (features, labels) = trainTestSplit[0]
            (vFeatures, vLabels) = trainTestSplit[1]
            (tFeatures, tLabels) = trainTestSplit[2]

            # --------------------------- THE TRAINING LOOP --------------------------- #
            index, i, vIndex, learningCurveInd, learningCurveTrn, learningCurveVal = 0, 0, 0, [], [], []

            # The training will run until it finishes all epochs, or until the validation accuracy exceeds 92% (early stopping).
            while i == 0 or (i in range(FLAGS.num_epochs) and learningCurveVal[len(learningCurveVal)-1] < 92):

                (bFeatures, bLabels) = _get_examples_batch(features, labels, index)
                (bvFeatures, bvLabels) = _get_examples_batch(vFeatures, vLabels, vIndex)
                [num_steps, loss, _] = sess.run([global_step_tensor, loss_tensor, train_op], feed_dict={features_tensor: bFeatures, labels_tensor: bLabels})
                print('Step %d: loss %g' % (num_steps, loss))

                result = sess.run(predictionTest, feed_dict={features_tensor: bFeatures, labels_tensor: bLabels})
                validate = sess.run(predictionTest, feed_dict={features_tensor: bvFeatures, labels_tensor: bvLabels})

                index += _BATCH_SIZE
                vIndex += _BATCH_SIZE

                # The 2 following IF statements are to reset the index when all the batches have been completed.
                # It also only validates at the end of each epoch, to reduce computation time.
                if index > len(features):
                    index = 0
                    learningCurveInd.append(i + 1)
                    learningCurveTrn.append(_test_model(result, bLabels, sess))
                    learningCurveVal.append(_test_model(validate, bvLabels, sess))
                    i += 1

                if vIndex > len(vFeatures):
                    vIndex = 0

            with open(os.getcwd() + "/TestingModel/LearningCurveFromTraining.txt", "wb") as fp:
                pickle.dump((learningCurveInd, learningCurveTrn, learningCurveVal), fp)
            saver.save(sess, os.getcwd() + "/TestingModel/TrainedModel")

        # --------------------------- THE TESTING LOOP --------------------------- #
        index, loop = 0, True

        # Restores the trained model
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + "/TestingModel/"))

        # Loops through all the test batches, performing only forward-passes.
        while loop:
            (bFeatures, bLabels) = _get_examples_batch(tFeatures, tLabels, index)
            if index != 0:
                testPredictions = np.concatenate((testPredictions, sess.run(predictionTest, feed_dict={features_tensor: bFeatures})))
            else:
                testPredictions = sess.run(predictionTest, feed_dict={features_tensor: bFeatures})
            index += _BATCH_SIZE
            if index >= len(tFeatures):
                loop = False

        # Finds the accuracy and prints confusion matrix.
        testAcc = _test_model(testPredictions, tLabels, sess, confM=True)

        # Plots the learning curve and accuracy comparison.
        plt.subplot(1, 1, 1)
        plt.title('Learning / Validation Curve (from when the imported model was trained)', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.plot(learningCurveInd, learningCurveTrn, lw=2, c='b', label='Training')
        plt.plot(learningCurveInd, learningCurveVal, lw=2, c='orange', label='Validation')
        plt.legend(loc='lower center')
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.subplot(1, 1, 1)
        plt.title('Accuracies (from when the imported model was trained)', fontsize=16)
        x = np.arange(3)
        bar = plt.bar(x, height=[learningCurveTrn[len(learningCurveTrn)-1], learningCurveVal[len(learningCurveVal)-1], testAcc])
        bar[0].set_color('royalblue')
        bar[1].set_color('orange')
        bar[2].set_color('firebrick')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(x, ['Train', 'Validate', 'Test'])
        plt.ylim([0, 100])
        plt.setp(bar, edgecolor="black", lw=2)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    tf.app.run()
