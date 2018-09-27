from __future__ import print_function

import zipfile
import numpy as np
import tensorflow as tf
import soundfile as sf
import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
    'num_batches', 20,
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

_NUM_CLASSES = 2


def _get_examples_batch():

    coffeeData, photoCopierData = [], []

    with zipfile.ZipFile("SoundFiles/coffeeMachine.zip", "r") as f:
        for name in f.namelist():
            coffeeData.append(vggish_input.wavfile_to_examples("SoundFiles/CoffeeMachine/" + name))
    with zipfile.ZipFile("SoundFiles/photoCopier.zip", "r") as f:
        for name in f.namelist():
            photoCopierData.append(vggish_input.wavfile_to_examples("SoundFiles/PhotoCopier/" + name))

    coffeeExamples = coffeeData[0]
    photoCopierExamples = photoCopierData[0]
    tFeatures = photoCopierData[0]


    for i, ex in enumerate(coffeeData):
        if i != 0:
            coffeeExamples = np.concatenate((coffeeExamples, ex))
    for i, ex in enumerate(photoCopierData):
        if i != 0:
            photoCopierExamples = np.concatenate((photoCopierExamples, ex))

    coffeeLabels = np.array([[1, 0]] * coffeeExamples.shape[0])
    photoCopierLabels = np.array([[0, 1]] * photoCopierExamples.shape[0])
    tLabels = np.array([[0, 1]])
    features = np.concatenate((coffeeExamples, photoCopierExamples))
    labels = np.concatenate((coffeeLabels, photoCopierLabels))
    split = [(features, labels), (tFeatures, tLabels)]

    return split


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope('mymodel'):
            # Add a fully connected layer with 100 units.
            num_units = 100
            fc = slim.fully_connected(embeddings, num_units)

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc, _NUM_CLASSES, activation_fn=None, scope='logits')
            logits = tf.sigmoid(logits, name='prediction')
            prediction = tf.round(logits)
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
                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=vggish_params.LEARNING_RATE,
                    epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)

        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # The training loop.
        for _ in range(FLAGS.num_batches):
            trainTestSplit = _get_examples_batch()
            (features, labels) = trainTestSplit[0]
            (tFeatures, tLabels) = trainTestSplit[1]
            [num_steps, loss, _] = sess.run(
                [global_step_tensor, loss_tensor, train_op],
                feed_dict={features_tensor: features, labels_tensor: labels})
            print('Step %d: loss %g' % (num_steps, loss))
        print(sess.run(prediction, feed_dict={features_tensor: tFeatures}))
        print("\nCorrect: ", tLabels)


if __name__ == '__main__':
    tf.app.run()
