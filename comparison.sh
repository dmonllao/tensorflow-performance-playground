#!/bin/bash

ACTIVATION='sigmoid'
NUM_EPOCH=30
N_HIDDEN=100
KEEP_PROB=0.9
OPTIMIZER='adam'

# MNIST data.
N_SAMPLES=60000
N_FEATURES=784
N_CLASSES=10

# Exponential decay between the start learning rate and the end learning rate.
# Better use the same value for both vars if using varying_batch_size as
# the required steps to apply the next decay are calculated using averages.
START_LEARNING_RATE=0.1
END_LEARNING_RATE=0.1

set -e

python clear_summaries.py

# Log increment from 100 to 10000.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='log_increase'

# Linear increment from 100 to 10000.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear_increase'

# Linear decrease from 10000 to 100.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear_decrease'

# Logarithmic decrease from 10000 to 100.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='log_decrease'

# Exponential increment from 100 to 10000.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='exp_increase'

# Exponential decrease from 10000 to 100.
python tfplayground/research/varying_batch_size.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=100 -maxb=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='exp_decrease'

exit 0
# Fixed batch size 100
python tfplayground/train.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=100 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB"

# Fixed to 10000.
python tfplayground/train.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=10000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB"

# Fixed batch size of all training set.
python tfplayground/train.py /home/davidm/Desktop/mnist_data/mnist_train.csv \
    /home/davidm/Desktop/mnist_data/mnist_test.csv \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=60000 -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB"
