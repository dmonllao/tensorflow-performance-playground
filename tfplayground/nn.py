import tensorflow as tf

def build_graph(n_samples, n_features, n_hidden, n_classes, x, y_, activation,
                start_lr, keep_prob=1, optimizer='gradientdescent',
                learning_rate_decay=0.9, test_data=False, normalize_input='',
                l2_regularization=0.):
    """Builds the computational graph without feeding any data in"""

    # Activation function.
    activation_function = get_activation_function(activation)

    # Input data info.
    with tf.name_scope('inputs'):
        tf.summary.histogram('x', x)

        if normalize_input != '':
            input_normalization = get_input_normalization(normalize_input)
            x = input_normalization(x, 0)
            tf.summary.histogram('x_normalized', x)

        shape = tf.shape(x)
        tf.summary.scalar('batch_size', shape[0])

    # Variables for computed stuff, we need to initialise them now.
    with tf.name_scope('weights'):

        test_accuracy = tf.Variable(0.0, dtype=tf.float32)

        W = []
        b = []

        # Input to first hidden layer.
        key = 'input-hidden_1'
        W.append(tf.Variable(
            tf.random_normal([n_features, n_hidden[0]], dtype=tf.float32)))
        tf.summary.histogram('W_' + key, W[0])

        b.append(tf.Variable(
                tf.random_normal([n_hidden[0]], dtype=tf.float32)))
        tf.summary.histogram('b_' + key, b[0])

        for i in range(1, len(n_hidden)):

            # +1 because of naming.
            n_hidden_layer = i + 1

            key = 'hidden_' + str(i) + '-hidden_' + str(n_hidden_layer)
            W.append(tf.Variable(
                tf.random_normal([n_hidden[i - 1], n_hidden[i]], dtype=tf.float32)))
            tf.summary.histogram('W_' + key, W[i])

            b.append(tf.Variable(
                tf.random_normal([n_hidden[i]], dtype=tf.float32)))
            tf.summary.histogram('b_' + key, b[i])


        # Last hidden to output layer.
        last_hidden = len(n_hidden) - 1
        key = 'hidden_' + str(last_hidden + 1) + '-output'
        W.append(tf.Variable(
                tf.random_normal([n_hidden[last_hidden], n_classes], dtype=tf.float32)))
        tf.summary.histogram('W_' + key, W[last_hidden])

        b.append(tf.Variable(
                tf.random_normal([n_classes], dtype=tf.float32)))
        tf.summary.histogram('b_' + key, b[last_hidden])

    # Predicted values and activations.
    with tf.name_scope('feed_forward'):

        if keep_prob < 1:
            x = tf.nn.dropout(x, keep_prob)

        predicted, activation, predicted_output, y = feed_forward(x, (W, b), activation_function)

        for i in range(len(predicted)):
            tf.summary.histogram('predicted_' + str(i + 1), predicted[str(i)])
            tf.summary.histogram('activation_' + str(i + 1), activation[str(i)])

        tf.summary.histogram('predicted_output', predicted_output)
        tf.summary.histogram('activation_output', y)

    # Cost function.
    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicted_output, labels=y_)
        tf.summary.scalar("loss", tf.reduce_mean(loss))
        if l2_regularization > 0.:
            l2_loss = tf.nn.l2_loss(W[-1])
            loss = tf.reduce_mean(loss + (l2_regularization * l2_loss))
            tf.summary.scalar("regularized_loss", loss)
        else:
            loss = tf.reduce_mean(loss)

    # Training and test accuracy.
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training_accuracy', accuracy)

        # Calculate the test dataset accuracy.
        if test_data is not False:
            test_x = tf.convert_to_tensor(test_data[0].values.tolist())
            test_y = tf.convert_to_tensor(test_data[1])
            _, _, test_probs, test_softmax = feed_forward(test_x, (W, b), activation_function)

            correct_prediction = tf.equal(tf.argmax(test_softmax, 1), tf.argmax(test_y, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)


    # Calculate decay_rate.
    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        decay_steps = tf.floor(tf.cast(n_samples / shape[0], tf.float32))
        tf.summary.scalar('decay_steps', decay_steps)
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

    predicted = {}
    activation = {}

    predicted['0'] = tf.add(tf.matmul(x, W[0]), b[0])
    activation['0'] = activation_function(predicted['0'])
    for i in range(1, len(W) - 1):
        predicted[str(i)] = tf.add(tf.matmul(activation[str(i - 1)], W[i]), b[i])
        activation[str(i)] = activation_function(predicted[str(i)])

    output_layer = len(W) - 1
    predicted_output = tf.add(
        tf.matmul(activation[str(output_layer - 1)], W[output_layer]),
        b[-1])
    activation_output = tf.nn.softmax(predicted_output, name='softmax-output')

    return predicted, activation, predicted_output, activation_output


def get_activation_function(name):
    return {
        'sigmoid': tf.sigmoid,
        'tanh': tf.tanh,
        'relu': tf.nn.relu
    }[name]


def get_input_normalization(name):
    return {
        'l2': tf.nn.l2_normalize
    }[name]


def get_optimizer(name, learning_rate):

    if name == 'gradientdescent':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

    raise ValueError('The provided optimizer is not valid')

