Tensorflow playground for comparing different algorithms performance.

To tune the neural network:

    vim main.py


To run it using your supervised learning dataset (no categorical features support).

    python main.py /path/to/training1.csv /path/to/training2.csv /path/to/testing.csv


To view the results:

    tensorboard --logdir=summaries
