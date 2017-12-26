import csv
import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf


def exp_multiplier(start, end, num_epochs):
    percent = float(end) / float(start)
    return np.power(percent, 1. / num_epochs)


def log_multiplier(start, end, num_epochs):
    percent = float(end) / float(start)
    return np.power(num_epochs, 1 / percent)


def calculate_lr_decay(start_lr, end_lr, batch_size,
                       n_samples, num_epochs):
    """Calculate approximately the optimal learning rate decay values"""

    # No decay.
    if start_lr == end_lr:
        return 1, batch_size

    # Set learning rate decay so that it is the end learning rate after num_epoch.
    learning_rate_decay = exp_multiplier(start_lr, end_lr, num_epochs)

    # The learning rate should decay after each epoch, otherwise the samples at
    # the end of the dataset have less weight.
    if batch_size is None:
        decay_steps = 1
    else:
        decay_steps = np.floor(n_samples / batch_size)

    print('Learning rate decay: ' + str(learning_rate_decay))
    print('Decay steps: ' + str(decay_steps))
    return learning_rate_decay, decay_steps


def parse_tfrecord_example(serialized_example):

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
    dataset = dataset.map(parse_tfrecord_example, num_parallel_calls=n_threads)

    dataset = dataset.repeat(num_epochs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    #dataset = dataset.shuffle(buffer_size=10000)

    return dataset


def get_n_neurons(n_features, n_classes, n_hidden, training_datasets):

    if n_features == None or n_classes == None:
        info = inputs.dataset_info(training_datasets)
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

    return n_features, n_classes, n_hidden


def dataset_info(training_datasets):
    info = {}
    with open(training_datasets[0], 'r') as f:
        vars = csv.DictReader(f)
        values = csv.DictReader(f)
        for i, var in enumerate(vars.fieldnames):
            info[var] = values.fieldnames[i]

    return info


def input_pipeline(training_datasets, n_features, skip_rows, batch_size, n_threads=1,
                   num_epochs=None):
    training_dataset_queue = tf.train.string_input_producer(
        training_datasets, num_epochs=num_epochs, shuffle=True)

    example_list = [read_pipeline(training_dataset_queue, n_features, skip_rows)
                    for _ in range(n_threads)]

    if batch_size is None:
        raise ValueError('batch_size var must have a value.')

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    return tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)


def read_pipeline(training_dataset_queue, n_features, skip_rows):
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


def test_data(filename, n_classes, label_first_column, skip_rows):

    df = pd.read_csv(filename, skiprows=skip_rows, dtype=np.float32)

    if label_first_column:
        # The label is the first column.
        y_one_hot = np.eye(n_classes)[df[df.columns[0]].astype(int)]
        features = df[df.columns[1:]].fillna(0)
    else:
        # The label is the last column.
        y_one_hot = np.eye(n_classes)[df[df.columns[-1]].astype(int)]
        features = df[df.columns[:-1]].fillna(0)

    return (features, y_one_hot)


def args_parser(description):

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('datasets', type=str, nargs='+',
                        help='Input files. All files but the last one will be used ' +
                              'for training. The last one will be used as test dataset.')
    parser.add_argument('--n_samples', '-m', dest='n_samples', type=int, required=True,
                        help='Used to calculate the number of training steps before a learning rate decay. No need to be exact')
    parser.add_argument('--n_features', '-n', dest='n_features', type=int, required=True,
                        help='Number of features')
    parser.add_argument('--n_classes', '-c', dest='n_classes', type=int, required=True,
                        help='Number of different labels')
    parser.add_argument('--skip_rows', '-s', dest='skip_rows', type=int, default=1,
                        help='Headers are not used, feel free to skip them.')
    parser.add_argument('--label_first_column', '-l', dest='label_first_column', action='store_true',
                        help='Set the flag if the label is in the first column instead of in the last one.')

    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=10000,
                        help='Whetever can be fit into the system memory.')
    parser.add_argument('--num_epochs', '-e', dest='num_epochs', type=int, default=100,
                        help='Number of times the algorithm will be trained using all provided training data.')
    parser.add_argument('--activation', '-a', dest='activation', type=str, default='tanh',
                        help='The activation function', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('--optimizer', '-o', dest='optimizer', type=str, default='gradientdescent',
                        help='Optimization method', choices=['gradientdescent', 'adam'])
    parser.add_argument('--n_hidden', '-nh', dest='n_hidden', type=int, help='Number of hidden layer neurons. ' +
                        'Automatically set to a value between n_features and n_classes if not value provided.')
    parser.add_argument('--keep_prob', '-d', dest='keep_prob', type=float, default=1,
                        help='Form dropout regularization')
    parser.add_argument('--start_learning_rate', '-slr', dest='start_lr', type=float, default=0.5,
                        help='Starting learning rate. It will decrease after each epoch')
    parser.add_argument('--end_learning_rate', '-elr', dest='end_lr', type=float, default=0.005,
                        help='The learning rate after num_epoch. It will gradually ' +
                             'decrease from start_learning_rate')

    return parser
