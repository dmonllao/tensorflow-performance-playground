import sys
import os
import time
import shutil
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import inputs
import nn

n_threads = 1
input_method = 'oldschool' # 'oldschool', 'dataset' or 'pipeline'.

#'--limiter', '-l', 'Set results limiter', choices=['time', 'epochs', 'testaccuracy']
#'--winner-metric', '-w', 'Sets the metric that determined the best classifier', choices=['time', 'testaccuracy', 'epochs']
#Can a teacher looking at logs be better than a machine detecting which students will not submit assignments on time?


############################################

parser = inputs.args_parser('Train the neural network.')

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

# Results logging.
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dir_path = (
    'batchsize_' + str(args.batch_size) + '-epoch_' + str(args.num_epochs) +
    '-learningrate_' + str(args.start_lr) +
    '-decay_' + str(lr_decay) + '-activation_' + args.activation +
    '-keepprob_' + str(args.keep_prob)
)
tensor_logdir = os.path.join(file_path, 'summaries', dir_path, str(time.time()))

# Load test data.
if test_dataset:
    test_data = inputs.test_data(test_dataset, n_classes, args.label_first_column, args.skip_rows)

# Inputs.
if input_method == 'pipeline':
    batches = inputs.input_pipeline(training_datasets, n_features, args.skip_rows,
                             args.batch_size, n_threads, num_epochs=args.num_epochs)
    x = batches[0]
    y_ = tf.one_hot(batches[1], n_classes)

elif input_method == 'oldschool':
    x = tf.placeholder(tf.float32, [None, n_features])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

elif input_method == 'dataset':
    dataset = inputs.get_tfrecord_dataset(training_datasets, args.batch_size, args.num_epochs,
                                   args.skip_rows, n_threads=n_threads)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    x = tf.reshape(next_element[0], [-1, n_features])
    y_ = tf.one_hot(next_element[1], n_classes)

# Build graph.
train_step, global_step, test_accuracy, model_vars = nn.build_graph(
    n_features, n_hidden, n_classes, x, y_, args.activation, args.start_lr,
    test_data=test_data, keep_prob=args.keep_prob, optimizer=args.optimizer,
    learning_rate_decay=lr_decay, decay_steps=decay_steps,
    normalize_input=args.normalize_input)

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

        except tf.errors.OutOfRangeError:
            print('Training finished')

        coord.request_stop()
        coord.join(threads)

    if input_method == 'oldschool':

        for i in range(args.num_epochs):
            for training_dataset in training_datasets:

                reader = pd.read_csv(training_dataset, chunksize=args.batch_size,
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
