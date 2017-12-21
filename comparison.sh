#!/bin/bash

ACTIVATION='sigmoid'
NUM_EPOCH=20
N_HIDDEN=100
KEEP_PROB=1
OPTIMIZER='gradientdescent'

# MNIST data.
N_SAMPLES=60000
N_FEATURES=784
N_CLASSES=10

START_LEARNING_RATE=0.1
END_LEARNING_RATE=0.001

set -e

python clear_summaries.py

# Log increment from 100 to 10000.
python incremental_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN -b=100 -eb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB" -o=$OPTIMIZER -i='logarithmic'

# Linear increment from 100 to 10000.
python incremental_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN -b=100 -eb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear'

# Linear decrease from 10000 to 100.
python incremental_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN -b=10000 -eb=100 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear'

# Log-shape decrease from 10000 to 100.
python incremental_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN -b=10000 -eb=100 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB" -o=$OPTIMIZER -i='exponential'

# Exponential increment from 100 to 10000.
python incremental_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN -b=100 -eb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB" -o=$OPTIMIZER -i='exponential'


# Fixed batch size 100
python train.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH -nh=$N_HIDDEN -b=100 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB"

# Fixed batch size 60000
python train.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH -nh=$N_HIDDEN -b=60000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB"

# Fixed to 10000.
python train.py /home/davidm/Desktop/mnist_data/mnist_train.csv /home/davidm/Desktop/mnist_data/mnist_test.csv -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH -nh=$N_HIDDEN -b=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" -d="$KEEP_PROB"
