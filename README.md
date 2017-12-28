Tensorflow playground for comparing different algorithms performance.

## Install

    git clone git://github.com/dmonllao/tensorflow-performance-playground.git
    cd tensorflow-performance-playground
    pip install -e .


## Usage

To view all configuration parameters:

    python tfplayground/train.py -h

To run it using your supervised learning dataset (no categorical features support).

    python tfplayground/train.py /path/to/training1.csv /path/to/training2.csv /path/to/testing.csv \
    --n_samples=60000 --n_features=784 --n_classes=10

    # Example using MNIST dataset.
    tar -xzvf "data/mnist/mnist.tar.gz" -C data/mnist/
    python tfplayground/train.py data/mnist/mnist_train.csv data/mnist/mnist_test.csv \
        --n_samples=60000 --n_features=784 --n_classes=10 --label_first_column

To view the results:

    tensorboard --logdir=summaries


## Compare results

Tune compare.sh script parameters (and executed scripts if necessary):

    vim compare.sh

Run compare script:

    ./compare.sh

Start tensorboard to see results as they are available:

    tensorboard --logdir=summaries
