import tensorflow as tf
import numpy as np
import os
import sys
from functools import partial


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W_conv = weight_variable([3, 3, 3, input_dim, output_dim])
            variable_summaries(W_conv)
        with tf.name_scope('biases'):
            b_conv = bias_variable([output_dim])
            variable_summaries(b_conv)
        with tf.name_scope('convolution'):
            preactivate = conv3d(input_tensor, W_conv) + b_conv
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.relu(preactivate)
        tf.summary.histogram('activations', activations)
        return activations


def pooling_layer(input_tensor, layer_name):
    with tf.name_scope(layer_name):
        pooled = tf.nn.max_pool3d(input_tensor, ksize=[1, 2, 2, 2, 1],
                                  strides=[1, 2, 2, 2, 1], padding='SAME')
        tf.summary.histogram('pooled', pooled)
    return pooled


def build_graph(x, y_, res):
    with tf.name_scope('reshape'):
        x_voxel = tf.reshape(x, [-1, res, res, res, 1])

    h_conv1 = conv_layer(x_voxel, 1, 8, "conv1")
    h_pool1 = pooling_layer(h_conv1, 'pool1')

    h_conv2 = conv_layer(h_pool1, 8, 14, "conv2")
    h_pool2 = pooling_layer(h_conv2, "pool2")

    h_conv3 = conv_layer(h_pool2, 14, 14, "conv3")
    h_pool3 = pooling_layer(h_conv3, "pool3")

    h_conv4 = conv_layer(h_pool3, 14, 20, "conv4")
    h_pool4 = pooling_layer(h_conv4, "pool4")

    h_conv5 = conv_layer(h_pool4, 20, 20, "conv5")
    h_pool5 = pooling_layer(h_conv5, "pool5")

    h_conv6 = conv_layer(h_pool5, 20, 26, "conv6")
    h_pool6 = pooling_layer(h_conv6, "pool6")

    h_conv7 = conv_layer(h_pool6, 26, 26, "conv7")
    h_conv8 = conv_layer(h_conv7, 26, 32, "conv8")
    h_conv9 = conv_layer(h_conv8, 32, 32, "conv9")
    h_conv10 = conv_layer(h_conv9, 32, 32, "conv10")

    h_pool10 = pooling_layer(h_conv10, 'pool10')

    # Reshape and fully connected
    with tf.name_scope('dropout'):
        h_pool10_flat = tf.reshape(h_pool10, [-1, 32 * 8 * 8 * 8])
        # To be able to turn it off during testing
        keep_prob = tf.placeholder(tf.float32)
        h_pool_drop = tf.nn.dropout(h_pool10_flat, keep_prob)

    with tf.name_scope('fully_connected'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([32 * 8 * 8 * 8, 512])
            variable_summaries(W_fc1)
        with tf.name_scope('biases'):
            b_fc1 = bias_variable([512])
            variable_summaries(b_fc1)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(h_pool_drop, W_fc1) + b_fc1
            tf.summary.histogram('pre_activations', preactivate)
        h_fc1 = tf.nn.relu(preactivate)
        tf.summary.histogram('activations', h_fc1)

    with tf.name_scope('readout'):
        with tf.name_scope('weights'):
            W_fc2 = weight_variable([512, 10])
            variable_summaries(W_fc2)
        with tf.name_scope('biases'):
            b_fc2 = bias_variable([10])
            variable_summaries(b_fc2)
        with tf.name_scope('Wx_plus_b'):
            y_readout = tf.matmul(h_fc1, W_fc2) + b_fc2
            tf.summary.histogram('pre_activations', y_readout)

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_readout)
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar("cross_entropy", cross_entropy)

    learning_rate = 0.1
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(
                tf.arg_max(y_readout, 1), tf.arg_max(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()

    return y_readout, merged


def main(_):
    # Import data
    data_list, labels_one_hot = read_data()

    # Create the model
    x = tf.placeholder(tf.float32, [None, vx_width * vx_height * vx_depth])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = build_graph(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_one_hot))

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()

    next_example, next_label = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):  # Number of epochs
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


def parse_function(width, height, depth, record):
    features = {
        'width': tf.FixedLenFeature((), tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'depth': tf.FixedLenFeature((), tf.int64),
        'data': tf.FixedLenFeature((width, height, depth), tf.float32),
        'label_one_hot': tf.VarLenFeature(tf.int64)
    }

    parsed_features = tf.parse_single_example(record, features)
    width = parsed_features['width']
    height = parsed_features['height']
    depth = parsed_features['depth']
    dense_labels = tf.sparse_tensor_to_dense(parsed_features['label_one_hot'])
    return parsed_features['data'], dense_labels


if __name__ == '__main__':
    batch_size = 10
    shuffle_size = 10000
    dir_name = sys.argv[1]
    res = sys.argv[2]

    training_records = os.path.join(dir_name, "test.tfrecord")

    dataset = tf.data.TFRecordDataset([training_records])
    dataset = dataset.map(partial(parse_function, res, res, res))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    features, labels = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        f_data, l_data = sess.run([features, labels])
        print(f_data, l_data)
