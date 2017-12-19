import tensorflow as tf
import numpy as np
import math
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

    h_conv1_1 = conv_layer(x_voxel, 1, 8, "conv1_1")
    h_conv1_2 = conv_layer(h_conv1_1, 8, 8, "conv1_2")
    h_pool_1 = pooling_layer(h_conv1_2, 'pool1')

    h_from_prev = h_pool_1

    for block in range(math.log(res, 2) - 3):
        input_dim = 8 + block * 6
        output_dim = input_dim + 6
        h_conv_1 = conv_layer(h_from_prev, input_dim,
                              output_dim, "conv" + str(block + 1) + "_1")
        h_conv_2 = conv_layer(h_conv_1, output_dim,
                              output_dim, "conv" + str(block + 1) + "_2")
        h_pool = pooling_layer(h_conv_2, 'pool1')

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
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cross_entropy)

    cat_predicted = tf.argmax(y_readout, 1)
    cat_label = tf.argmax(y_, 0)
    with tf.name_scope('accuracy'):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(cat_predicted, cat_label)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()

    return train_step, accuracy, merged, cat_predicted, cat_label, keep_prob


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


# if __name__ == '__main__':
def main():
    batch_size = 10
    shuffle_size = 10000
    dir_name = sys.argv[1]
    res = int(sys.argv[2])
    num_epochs = 4

    training_records = os.path.join(dir_name, "training.tfrecord")
    test_records = os.path.join(dir_name, "test.tfrecord")

    dataset = tf.data.TFRecordDataset([training_records])
    dataset = dataset.map(partial(parse_function, res, res, res))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size)

    test_dataset = tf.data.TFRecordDataset([test_records])

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, dataset.output_types, dataset.output_shapes)

    next_element, next_label = iterator.get_next()

    training_iterator = dataset.make_initializable_iterator()
    validation_iterator = test_dataset.make_initializable_iterator()

    train_step, merged, accuracy, predicted, label, keep_prob = build_graph(
        next_element, next_label, res)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            os.path.join(dir_name, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(dir_name, "test"), sess.graph)

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        tf.global_variables_initializer().run()

        for epoch in range(num_epochs):
            sess.run(training_iterator.initializer)
            step = -1
            print("Epoch: " + str(epoch))
            while True:
                print(" - Step: " + str(step))
                try:
                    if step % 10 == 0:  # Record summaries and test-set accuracy
                        step += 1
                        sess.run(validation_iterator.initializer)
                        # Run the whole thing
                        while True:
                            try:
                                summary, acc = sess.run([merged, accuracy], feed_dict={
                                                        handle: validation_handle, keep_prob: 1.0})
                                print('Accuracy at step %s: %s' % (step, acc))
                                # TODO: WRITE OUT
                                test_writer.add_summary(
                                    summary, epoch * 10000 + step)
                            except tf.errors.OutOfRangeError:
                                break
                    else:  # Record train set data summaries and train
                        step += 1
                        if step % 100 == 99:  # Record execution stats
                            run_options = tf.RunOptions(
                                trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, _ = sess.run([merged, train_step],
                                                  feed_dict={handle: training_handle, keep_prob: 0.5})
                            train_writer.add_run_metadata(
                                run_metadata, 'step%10d' % epoch * 10000 + step)
                            train_writer.add_summary(
                                summary, epoch * 10000 + step)
                        else:  # Record a summary
                            summary, _ = sess.run([merged, train_step], feed_dict={
                                                  handle: training_handle, keep_prob: 0.5})
                            train_writer.add_summary(
                                summary, epoch * 10000 + step)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    main()
