Tensorflow playground for comparing different algorithms performance.

To view all configuration parameters:

    python train.py -m

To run it using your supervised learning dataset (no categorical features support).

    python train.py /path/to/training1.csv /path/to/training2.csv /path/to/testing.csv \
    --n_samples=60000 --n_features=784 --n_classes=10


To view the results:

    tensorboard --logdir=summaries
