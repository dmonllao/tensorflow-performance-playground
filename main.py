import sys
import csv
import os
import time
import shutil
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf


# INPUT FILES DATA ###########################

# Headers are not used, feel free to include them here.
skip_rows = 3

# Used to calculate the number of training steps before a learning rate decay.
total_training_samples = 270000 # No need to be an exact value.

# Whether the label is the last column or the first one.
label_last_column = True

n_features = None # 'nfeatures' key

n_classes = None # 'targetclasses' key

# HYPERPARAMETERS ############################

# Whetever can be fit into the system memory.
batch_size = 100000 # None for no mini-batching.
num_epochs = 10

# Dropout regularisation.
keep_prob = 1

# Learning rate tuning. Decay calculated from start - end difference.
starting_learning_rate = 0.5
ending_learning_rate = 0.005

# Activation function.
activation = tf.sigmoid

# None to set it automatically based on the first two rows values: dataset_info()
n_hidden = None  # Automatically set to a value between n_features and n_classes

# OTHER STUFF ################################

n_threads = 1
input_method = 'oldschool' # 'oldschool', 'dataset' or 'pipeline'.

##############################################


def calculate_lr_decay(starting_learning_rate, ending_learning_rate,
                       n_samples, num_epochs):
    """Calculate approximately the optimal learning rate decay values"""

    # Set learning rate decay so that it is ending_learning_rate after num_epoch.
    percent_decrease = float(ending_learning_rate) / float(starting_learning_rate)
    learning_rate_decay = np.round(np.power(percent_decrease, 1. / num_epochs), 3)

    # The learning rate should decay after each epoch, otherwise the samples at
    # the end of the dataset have less weight.
    if batch_size is None:
        decay_steps = 1
    else:
        decay_steps = np.floor(n_samples / batch_size)

    print('Learning rate decay: ' + str(learning_rate_decay))
    print('Decay steps: ' + str(decay_steps))
    return learning_rate_decay, decay_steps


def parse_example(serialized_example):

    features = {
        "features": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(serialized_example, features)
    return parsed_features['features'], parsed_features['label']


def get_tfrecord_dataset(training_datasets, batch_size, num_epochs, skip_rows,
                         n_threads=None):

    tfrecord_files = []
    for training_dataset in training_datasets:
        filename, ext = os.path.splitext(training_dataset)
        output_file = filename + "_" + ext[1:] + ".tfrecords"

        # Skip already converted files.
        if os.path.isfile(output_file) == False:

            print('Generating tfrecord file...')
            writer = tf.python_io.TFRecordWriter(output_file)

            # Read in chunks to avoid memory problems.
            reader = pd.read_csv(training_dataset, chunksize=batch_size,
                                    skiprows=skip_rows,
                                    dtype=np.float32)
            # Only 1 chunk When batch_size = None, let's convert it to an iterable.
            if type(reader) is not pd.io.parsers.TextFileReader:
                reader = [reader]

            for df in reader:
                df = df.fillna(0.0)
                for i, row in df.iterrows():
                    features = row[:-1].tolist()
                    label = int(row[-1])
                    example = tf.train.Example()
                    example.features.feature["features"].float_list.value.extend(features)
                    example.features.feature["label"].int64_list.value.append(label)
                    writer.write(example.SerializeToString())

            writer.close()

        tfrecord_files.append(output_file)

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_example, num_parallel_calls=n_threads)

    dataset = dataset.repeat(num_epochs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    #dataset = dataset.shuffle(buffer_size=10000)

    return dataset


def dataset_info(training_datasets):
    info = {}
    with open(training_datasets[0], 'r') as f:
        vars = csv.DictReader(f)
        values = csv.DictReader(f)
        for i, var in enumerate(vars.fieldnames):
            info[var] = values.fieldnames[i]

    return info


def input_pipeline(training_datasets, n_features, batch_size, n_threads=1,
                   num_epochs=None):
    training_dataset_queue = tf.train.string_input_producer(
        training_datasets, num_epochs=num_epochs, shuffle=True)

    example_list = [read_pipeline(training_dataset_queue, n_features)
                    for _ in range(n_threads)]

    if batch_size is None:
        raise ValueError('batch_size var must have a value.')

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    return tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)


def read_pipeline(training_dataset_queue, n_features):
    reader = tf.TextLineReader(skip_header_lines=skip_rows)
    key, record_string = reader.read(training_dataset_queue)

    record_defaults = []
    for i in range(n_features):
        record_defaults.append([0.0])
    record_defaults.append([0])

    rows = tf.decode_csv(record_string, record_defaults)
    example = rows[0:n_features]
    label = rows[n_features]

    # TODO Preprocessing
    return example, label


def test_data(filename, n_classes, label_last_column):

    # -1 to get the headers.
    df = pd.read_csv(filename, skiprows=skip_rows, dtype=np.float32)

    if label_last_column:
        # The label is the last column.
        y_one_hot = np.eye(n_classes)[df[df.columns[-1]].astype(int)]
        features = df[df.columns[:-1]].fillna(0)
    else:
        # The label is the first column.
        y_one_hot = np.eye(n_classes)[df[df.columns[0]].astype(int)]
        features = df[df.columns[1:]].fillna(0)

    return (features, y_one_hot)


def feed_forward(x, model_vars, activation):
    """Single hidden layer feed forward nn using softmax."""

    W = model_vars[0]
    b = model_vars[1]

    hidden = activation(tf.matmul(x, W['input-hidden']) + b['input-hidden'],
                     name='activation-function')

    linear_values = tf.matmul(hidden, W['hidden-output']) + b['hidden-output']

    return linear_values, tf.nn.softmax(linear_values)


def build_nn_graph(n_features, n_hidden, n_classes, x, y_, activation, keep_prob=1,
                   learning_rate_decay=0.9, test_data=False, decay_steps=1000):
    """Builds the computational graph without feeding any data in"""

    # Variables for computed stuff, we need to initialise them now.
    with tf.name_scope('initialise-vars'):

        test_accuracy = tf.Variable(0.0, dtype=tf.float32)

        W = {
            'input-hidden': tf.Variable(
                tf.random_normal([n_features, n_hidden], dtype=tf.float32),
                name='input-to-hidden-weights'
            ),
            'hidden-output': tf.Variable(
                tf.random_normal([n_hidden, n_classes], dtype=tf.float32),
                name='hidden-to-output-weights'
            ),
        }

        b = {
            'input-hidden': tf.Variable(
                tf.random_normal([n_hidden], dtype=tf.float32),
                name='hidden-bias'
            ),
            'hidden-output': tf.Variable(
                tf.random_normal([n_classes], dtype=tf.float32),
                name='output-bias'
            ),
        }

    # Predicted y.
    with tf.name_scope('loss'):

        x = tf.nn.dropout(x, keep_prob)
        linear_values, y = feed_forward(x, (W, b), activation)
        tf.summary.histogram('predicted_values', linear_values)
        tf.summary.histogram('activations', y)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=linear_values, labels=y_)
        )
        tf.summary.scalar("loss", loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training_accuracy', accuracy)

        # Calculate the test dataset accuracy.
        if test_data is not False:
            test_probs, test_softmax = feed_forward(test_data[0], (W, b), activation)
            correct_prediction = tf.equal(tf.argmax(test_softmax, 1), tf.argmax(test_data[1], 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)


    # Calculate decay_rate.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starting_learning_rate,
                                               global_step, decay_steps,
                                               learning_rate_decay,
                                               staircase=False)
    tf.summary.scalar("learning_rate", learning_rate)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate
    ).minimize(loss, global_step=global_step)

    return train_step, global_step, test_accuracy, (W, b)


############################################

parser = argparse.ArgumentParser(description='Train the neural network using ' +
                                              'the provided setup.')
parser.add_argument('datasets', type=str, nargs='+',
                    help='Input files. All files but the last one will be used ' +
                          'for training. The last one will be used as test dataset.')

args = parser.parse_args()
datasets = args.datasets

if len(args.datasets) < 2:
    print('It is recommended to also provide a test dataset')
else:
    # The last item is the test set.
    test_dataset = datasets.pop()

training_datasets = datasets


start_time = time.clock()

if n_features == None or n_classes == None:
    info = dataset_info(training_datasets)
    if n_features == None:
        if info.get('nfeatures') == None:
            print('No n_features value has been provided')
            sys.exit()
        n_features = int(info.get('nfeatures'))

    if n_classes == None:
        if info.get('targetclasses') == False:
            print('No n_classes value has been provided')
            sys.exit()
        classes = eval(info.get('targetclasses'))
        n_classes = len(classes)

if n_hidden == None:
    n_hidden = max(int((n_features - n_classes) / 2), 2)

# Calculate learning rate decay.
lr_decay, decay_steps = calculate_lr_decay(starting_learning_rate,
                                           ending_learning_rate,
                                           total_training_samples,
                                           num_epochs)

# Results logging.
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = (
    'batchsize_' + str(batch_size) + '-epoch_' + str(num_epochs) +
    '-learningrate_' + str(starting_learning_rate) +
    '-decay_' + str(lr_decay) + '-activation_' + activation.__name__
)
tensor_logdir = os.path.join(file_path, 'summaries', dir_path, str(time.time()))

# Load test data.
if test_dataset:
    test_data = test_data(test_dataset, n_classes, label_last_column)

# Inputs.
if input_method == 'pipeline':
    batches = input_pipeline(training_datasets, n_features,
                             batch_size, n_threads, num_epochs=num_epochs)
    x = batches[0]
    y_ = tf.one_hot(batches[1], n_classes)

elif input_method == 'oldschool':
    x = tf.placeholder(tf.float32, [None, n_features])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

elif input_method == 'dataset':
    dataset = get_tfrecord_dataset(training_datasets, batch_size, num_epochs,
                                   skip_rows, n_threads=n_threads)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    x = tf.reshape(next_element[0], [-1, n_features])
    y_ = tf.one_hot(next_element[1], n_classes)

# Build graph.
train_step, global_step, test_accuracy, model_vars = build_nn_graph(
    n_features, n_hidden, n_classes,
    x, y_, activation=activation, test_data=test_data, keep_prob=keep_prob,
    learning_rate_decay=lr_decay, decay_steps=decay_steps)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    init = tf.local_variables_initializer()
    sess.run(init)

    file_writer = tf.summary.FileWriter(tensor_logdir, sess.graph)
    merged = tf.summary.merge_all()

    if input_method == 'pipeline' or input_method == 'dataset':

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                _, summary = sess.run([train_step, merged])
                global_step_value = sess.run(global_step)
                file_writer.add_summary(summary, global_step_value)

                # Output test dataset accuracy.
                if global_step_value % 10 == 0:
                    print('Test accuracy: ' + str(sess.run(test_accuracy)))


        except tf.errors.OutOfRangeError:
            print('Training finished')

        coord.request_stop()
        coord.join(threads)

    if input_method == 'oldschool':

        for i in range(num_epochs):
            for training_dataset in training_datasets:

                # -1 to get the headers.
                reader = pd.read_csv(training_dataset, chunksize=batch_size,
                                        skiprows=skip_rows,
                                        dtype=np.float32)

                # Only 1 chunk When batch_size = None, let's convert it to an iterable.
                if type(reader) is not pd.io.parsers.TextFileReader:
                    reader = [reader]

                for df in reader:

                    if label_last_column:
                        # The label is the last column.
                        y_one_hot = np.eye(n_classes)[df[df.columns[-1]].astype(int)]
                        features = df[df.columns[:-1]].fillna(0)
                    else:
                        # The label is the first column.
                        y_one_hot = np.eye(n_classes)[df[df.columns[0]].astype(int)]
                        features = df[df.columns[1:]].fillna(0)

                    # Run 1 training iteration.
                    _, summary = sess.run([train_step, merged], {
                        x: features,
                        y_: y_one_hot
                    })

                    # Report about loss, accuracies...
                    global_step_value = sess.run(global_step)
                    file_writer.add_summary(summary, global_step_value)

                    # Output test dataset accuracy.
                    if global_step_value % 10 == 0:
                        print('Test accuracy: ' + str(sess.run(test_accuracy)))

end_time = time.clock()

print('Time elapsed: ' + str(end_time - start_time))
