import sys
import csv
import os
import time
import math
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

import inputs
import nn

n_threads = 1
input_method = 'oldschool' # 'oldschool', 'dataset' or 'pipeline'.


def batch_logarithmic_sizes(start_batch_size, end_batch_size, num_epochs):
    """"Logarithmic scale of batch sizes of size num_epochs"""

    multiplier = inputs.logarithmic_multiplier(start_batch_size, end_batch_size, num_epochs)
    print('Batch logarithmic multiplier: ' + str(multiplier))

    sizes = [start_batch_size]
    for i in range(1, num_epochs):
        log = math.log(i + 1, multiplier)
        sizes.append(int(start_batch_size * log))

    print(sizes)
    return sizes


def batch_exponential_sizes(start_batch_size, end_batch_size, num_epochs):
    """"Exponential scale of batch sizes of size num_epochs"""

    multiplier = inputs.exponential_multiplier(start_batch_size, end_batch_size, num_epochs - 1)
    print('Batch exponential multiplier: ' + str(multiplier))

    sizes = [start_batch_size]
    for i in range(1, num_epochs):
        exp = math.pow(multiplier, i)
        sizes.append(int(start_batch_size * exp))

    print(sizes)
    return sizes


def batch_linear_sizes(start_batch_size, end_batch_size, num_epochs):
    """"Linear scale of batch sizes of size num_epochs"""

    addition = (end_batch_size - start_batch_size) / (num_epochs - 1)
    print('Batch linear addition: ' + str(addition))
    sizes = [start_batch_size]
    for i in range(1, num_epochs):
        sizes.append(start_batch_size + (addition * i))

    print(sizes)
    return sizes


############################################

parser = inputs.args_parser('Train the neural network using an incremental batch size.')
parser.add_argument('--end_batch_size', '-eb', dest='end_batch_size', type=int,
                    default=10000, help='Whetever can be fit into the system memory.')
parser.add_argument('--increment', '-i', dest='increment_method', type=str,
                    choices=['logarithmic', 'exponential', 'linear'],
                    default='linear', help='How should the batch size increase')

args = parser.parse_args()
print(args)

datasets = args.datasets

if len(args.datasets) < 2:
    print('It is recommended to also provide a test dataset')
else:
    # The last item is the test set.
    test_dataset = datasets.pop()

training_datasets = datasets

start_time = time.clock()

# Network architecture.
n_features, n_classes, n_hidden = inputs.get_n_neurons(args.n_features,
    args.n_classes, args.n_hidden, training_datasets)

# Calculate learning rate decay.
lr_decay, decay_steps = inputs.calculate_lr_decay(args.start_lr,
                                           args.end_lr,
                                           args.batch_size,
                                           args.n_samples,
                                           args.num_epochs)


# Batch size increment method.
increment_method = 'batch_' + args.increment_method + '_sizes'
batch_sizes = locals()[increment_method](args.batch_size, args.end_batch_size, args.num_epochs)

# Results logging.
file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = (
    'increment_' + str(args.increment_method) +
    '-batchsize_' + str(args.batch_size) + '-endbatchsize_' + str(args.end_batch_size) +
    '-epoch_' + str(args.num_epochs) +
    '-learningrate_' + str(args.start_lr) +
    '-decay_' + str(lr_decay) + '-activation_' + args.activation +
    '-keepprob_' + str(args.keep_prob)
)
tensor_logdir = os.path.join(file_path, 'summaries', dir_path, str(time.time()))

# Load test data.
if test_dataset:
    test_data = inputs.test_data(test_dataset, n_classes, args.label_first_column, args.skip_rows)

x = tf.placeholder(tf.float32, [None, n_features])
y_ = tf.placeholder(tf.float32, [None, n_classes])

# Build graph.
train_step, global_step, test_accuracy, model_vars = nn.build_graph(
    n_features, n_hidden, n_classes, x, y_, args.activation, args.start_lr,
    test_data=test_data, keep_prob=args.keep_prob, optimizer=args.optimizer,
    learning_rate_decay=lr_decay, decay_steps=decay_steps)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    init = tf.local_variables_initializer()
    sess.run(init)

    file_writer = tf.summary.FileWriter(tensor_logdir, sess.graph)
    merged = tf.summary.merge_all()

    for i in range(args.num_epochs):
        for training_dataset in training_datasets:

            batch_size = batch_sizes[i]
            print('Batch size: ' + str(batch_size))

            reader = pd.read_csv(training_dataset, chunksize=batch_size,
                                    skiprows=args.skip_rows,
                                    dtype=np.float32)

            # Only 1 chunk When batch_size = None, let's convert it to an iterable.
            if type(reader) is not pd.io.parsers.TextFileReader:
                reader = [reader]

            for df in reader:

                if args.label_first_column:
                    # The label is the first column.
                    y_one_hot = np.eye(n_classes)[df[df.columns[0]].astype(int)]
                    features = df[df.columns[1:]].fillna(0)
                else:
                    # The label is the last column.
                    y_one_hot = np.eye(n_classes)[df[df.columns[-1]].astype(int)]
                    features = df[df.columns[:-1]].fillna(0)

                # Run 1 training iteration.
                _, summary = sess.run([train_step, merged], {
                    x: features,
                    y_: y_one_hot
                })

                # Report about loss, accuracies...
                global_step_value = sess.run(global_step)
                file_writer.add_summary(summary, global_step_value)

    print('Test accuracy: ' + str(sess.run(test_accuracy)))

end_time = time.clock()

print('Time elapsed: ' + str(end_time - start_time))
