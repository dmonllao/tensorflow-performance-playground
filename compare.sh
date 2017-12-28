#!/bin/bash

# Use the included MNIST dataset by default.
TRAIN_FILE="data/mnist/mnist_train.csv"
TEST_FILE="data/mnist/mnist_test.csv"

# MNIST data.
N_SAMPLES=60000
N_FEATURES=784
N_CLASSES=10

ACTIVATION='tanh'
NUM_EPOCH=30
N_HIDDEN=100
KEEP_PROB=0.9 # Use 1. for no dropout regularization.
L2_REG_TERM=0.01 # Use 0. for no l2 regularization.
OPTIMIZER='adam'
INPUT_NORM='l2'

MIN_BATCH_SIZE=100
MAX_BATCH_SIZE=10000
# Only used with static batch size.
MID_BATCH_SIZE=1000

# Exponential decay between the start learning rate and the end learning rate.
# Better use the same value for both vars if using varying_batch_size as
# the required steps to apply the next decay are calculated using averages.
START_LEARNING_RATE=0.1
END_LEARNING_RATE=0.1

set -e

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    if [ "$TRAIN_FILE" == "data/mnist/mnist_train.csv" ]; then
        # Extract the included MNIST dataset.
        tar -xzvf "data/mnist/mnist.tar.gz" -C data/mnist/
    else
        echo "Training or test file do not exist."
        exit 1
    fi
fi

python clear_summaries.py 1> /dev/null

# Log increment from $MIN_BATCH_SIZE to $MAX_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='log_increase' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Linear increment from $MIN_BATCH_SIZE to $MAX_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear_increase' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Linear decrease from $MAX_BATCH_SIZE to $MIN_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='linear_decrease' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Logarithmic decrease from $MAX_BATCH_SIZE to $MIN_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='log_decrease' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Exponential increment from $MIN_BATCH_SIZE to $MAX_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='exp_increase' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Exponential decrease from $MAX_BATCH_SIZE to $MIN_BATCH_SIZE.
python tfplayground/research/varying_batch_size.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -e=$NUM_EPOCH -nh=$N_HIDDEN \
    -minb=$MIN_BATCH_SIZE -maxb=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" -o=$OPTIMIZER -i='exp_decrease' --norm="$INPUT_NORM" \
    -l2="$L2_REG_TERM"

# Fixed batch size $MIN_BATCH_SIZE
python tfplayground/train.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=$MIN_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" --norm="$INPUT_NORM" -l2="$L2_REG_TERM"

# Fixed batch size $MID_BATCH_SIZE
python tfplayground/train.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=$MID_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" --norm="$INPUT_NORM" -l2="$L2_REG_TERM"

# Fixed to $MAX_BATCH_SIZE.
python tfplayground/train.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=$MAX_BATCH_SIZE -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" --norm="$INPUT_NORM" -l2="$L2_REG_TERM"

# Fixed batch size of all training set.
python tfplayground/train.py "$TRAIN_FILE" "$TEST_FILE" \
    -m=$N_SAMPLES -n=$N_FEATURES -c=$N_CLASSES -l -o=$OPTIMIZER -e=$NUM_EPOCH \
    -nh=$N_HIDDEN -b=$N_SAMPLES -slr=$START_LEARNING_RATE -elr=$END_LEARNING_RATE -a="$ACTIVATION" \
    -d="$KEEP_PROB" --norm="$INPUT_NORM" -l2="$L2_REG_TERM"
