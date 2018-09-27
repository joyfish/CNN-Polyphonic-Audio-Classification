from __future__ import print_function
import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import vggish_params
import vggish_slim
import read_in_dataset
from tensorflow.python.ops import nn_ops


flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_epochs', 250,
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

_NUM_CLASSES = 20
_BATCH_SIZE = 650
# np.set_printoptions(threshold=np.nan)

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


def _test_model(predictions, vLabels, sess, classes):
    with open("/cs/scratch/pmh20/predictions.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(predictions)

    maxes = np.argmax(predictions, axis=1)
    classification = predictions.astype(int)

    for i, _ in enumerate(classification):
        for j, _ in enumerate(classification[i]):
            classification[i][j] = 0
            classification[i][maxes[i]] = 1
    # print("\nClassifications:\n", predictions, "\n")

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
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.title('Confusion Matrix', fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(_NUM_CLASSES)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    normalise = True
    if normalise:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalise else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center", verticalalignment="center", fontsize=8,
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    print("\nActual Labels:   \n", correctLabels, "\n")
    print("\nPredicted Labels:\n", predictedLabels, "\n")
    print("\nComparison:\n", comparison, "\n")
    accuracy = (sum(comparison) / len(comparison)) * 100.0
    print("\nValidation Accuracy: ", accuracy, "%\n\n")
    #plt.show()


def main(_):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
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
                # print(xent.get_shape())
                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=vggish_params.LEARNING_RATE,
                    epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

        # Initialize all variables in the model, and then load the pre-trained VGGish checkpoint.
        # writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)

        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        tf.summary.scalar('loss', loss_tensor)
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        trainTestSplit = read_in_dataset._get_train_test_split_new_mixed(2)
        classes = trainTestSplit[0]
        (features, labels) = trainTestSplit[1]
        (vFeatures, vLabels) = trainTestSplit[2]
        try: (tFeatures, tLabels) = trainTestSplit[3]
        except: 1
        try: (split1Features, split2Features) = trainTestSplit[4]
        except: 1
        try: (split3Features, split4Features) = trainTestSplit[5]
        except: 1

        # merged = tf.summary.merge_all()

        # The training loop.
        index = 0
        for i in range(FLAGS.num_epochs):
            (bFeatures, bLabels) = _get_examples_batch(features, labels, index)
            [num_steps, loss, _] = sess.run([global_step_tensor, loss_tensor, train_op], feed_dict={features_tensor: bFeatures, labels_tensor: bLabels})
            print('Step %d: loss %g' % (num_steps, loss))
            # result = sess.run(feed_dict={features_tensor: bFeatures, labels_tensor: bLabels})
            # writer.add_summary(result, i)
            index += _BATCH_SIZE
            if index > len(features):
                index = 0

        # The validation loop
        print("\n\n-----VALIDATION-----\n")
        index, loop = 0, True
        while loop:
            (bFeatures, bLabels) = _get_examples_batch(vFeatures, vLabels, index)
            if index != 0:
                testPredictions = np.concatenate((testPredictions, sess.run(predictionTest, feed_dict={features_tensor: bFeatures})))
            else:
                testPredictions = sess.run(predictionTest, feed_dict={features_tensor: bFeatures})
            index += _BATCH_SIZE
            if index >= len(vFeatures):
                loop = False
        _test_model(testPredictions, vLabels, sess, classes)

        if len(trainTestSplit) > 3:
            # The testing loop
            print("\n\n-----TESTING (MIXED)-----\n")
            accuracies = []
            for j in range(2):
                if j == 0:
                    a = split1Features
                    b = split2Features
                else:
                    a = split3Features
                    b = split4Features
                for i, (spl1, spl2) in enumerate(zip(a, b)):
                    try: pred1 = sess.run(predictionTest, feed_dict={features_tensor: spl1})
                    except: pred1 = pred1
                    try: pred2 = sess.run(predictionTest, feed_dict={features_tensor: spl2})
                    except: pred2 = pred2
                    maxes1 = np.argmax(pred1, axis=1)
                    maxes2 = np.argmax(pred2, axis=1)
                    # print(maxes1)
                    # print(maxes2)
                    majVote1 = np.argmax(np.bincount(maxes1))
                    majVote2 = np.argmax(np.bincount(maxes2))
                    mixTestPredictions = [0] * _NUM_CLASSES
                    mixTestPredictions[majVote1] += 1
                    mixTestPredictions[majVote2] += 1
                    acc = _test_model_mixed(mixTestPredictions, tLabels[i])
                    if i < 20:
                        print(i, ":", acc, "-", mixTestPredictions)
                    accuracies.append(acc)
                    # print(i, ":", acc)
                overallAcc = sum(accuracies) / float(len(accuracies))
                print("\n\nAccuracy with mixed events (", j, "):", overallAcc, "%\n\n")


if __name__ == '__main__':
    tf.app.run()
