import tensorflow as tf


def build_graph(n_features, n_hidden, n_classes, x, y_, activation, start_lr,
                keep_prob=1, optimizer='gradientdescent', learning_rate_decay=0.9,
                test_data=False, decay_steps=1000):
    """Builds the computational graph without feeding any data in"""

    # Activation function.
    activation_function = get_activation_function(activation)

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
        linear_values, y = feed_forward(x, (W, b), activation_function)
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
            test_probs, test_softmax = feed_forward(test_data[0], (W, b), activation_function)
            correct_prediction = tf.equal(tf.argmax(test_softmax, 1), tf.argmax(test_data[1], 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)


    # Calculate decay_rate.
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_lr,
                                               global_step, decay_steps,
                                               learning_rate_decay,
                                               staircase=False)
    tf.summary.scalar("learning_rate", learning_rate)

    optimizer = get_optimizer(optimizer, learning_rate)
    train_step = optimizer.minimize(loss, global_step=global_step)

    return train_step, global_step, test_accuracy, (W, b)


def feed_forward(x, model_vars, activation_function):
    """Single hidden layer feed forward nn using softmax."""

    W = model_vars[0]
    b = model_vars[1]

    hidden = activation_function(tf.matmul(x, W['input-hidden']) + b['input-hidden'],
                     name='activation-function')

    linear_values = tf.matmul(hidden, W['hidden-output']) + b['hidden-output']

    return linear_values, tf.nn.softmax(linear_values)


def get_activation_function(name):
    return {
        'sigmoid': tf.sigmoid,
        'tanh': tf.tanh,
        'relu': tf.nn.relu
    }[name]

def get_optimizer(name, learning_rate):

    if name == 'gradientdescent':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

    raise ValueError('The provided optimizer is not valid')

