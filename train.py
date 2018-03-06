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


def parse_function(width, height, depth, cats, record):
    features = {
        'width': tf.FixedLenFeature((), tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'depth': tf.FixedLenFeature((), tf.int64),
        'data': tf.FixedLenFeature((width, height, depth), tf.float32),
        'label_one_hot': tf.FixedLenFeature((cats), tf.int64)  # TODO
    }

    parsed_features = tf.parse_single_example(record, features)
    width = parsed_features['width']
    height = parsed_features['height']
    depth = parsed_features['depth']
    #dense_labels = tf.sparse_tensor_to_dense(parsed_features['label_one_hot'])
    # print "parsing: "
    # print dense_labels.get_shape()
    #dense_labels = tf.reshape(dense_labels)
    return parsed_features['data'], parsed_features['label_one_hot']


def train(dir_name, res):
    batch_size = 32
    shuffle_size = 10000
    num_epochs = 12

    training_records = os.path.join(dir_name, "training.tfrecord")
    test_records = os.path.join(dir_name, "test.tfrecord")

    num_cats = sum(1 for line in open(os.path.join(dir_name, "labels.txt")))
    print("Found " + str(num_cats) + " categories")

    dataset = tf.data.TFRecordDataset([training_records])
    dataset = dataset.map(partial(parse_function, res, res, res, num_cats))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size)

    test_dataset = tf.data.TFRecordDataset([test_records])
    test_dataset = test_dataset.map(
        partial(parse_function, res, res, res, num_cats))
    test_dataset = test_dataset.shuffle(shuffle_size)
    test_dataset = test_dataset.batch(64)

    handle = tf.placeholder(tf.string, shape=[], name="handle")
    iterator = tf.data.Iterator.from_string_handle(
        handle, dataset.output_types, dataset.output_shapes)

    next_element, next_label = iterator.get_next()

    training_iterator = dataset.make_initializable_iterator()
    validation_iterator = test_dataset.make_initializable_iterator()

    sess = tf.InteractiveSession()

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    x = tf.placeholder_with_default(next_element, shape=None, name="x_input")
    y_ = next_label

    # BUILD GRAPH
    with tf.name_scope('reshape'):
        x_voxel = tf.reshape(x, [-1, res, res, res, 1])

    h_conv1_1 = conv_layer(x_voxel, 1, 8, "conv1_1")
    h_conv1_2 = conv_layer(h_conv1_1, 8, 8, "conv1_2")
    h_pool_1 = pooling_layer(h_conv1_2, 'pool1')

    h_from_prev = h_pool_1

    output_dim = 8
    for block in range(int(math.log(res, 2)) - 4):
        input_dim = 8 + block * 6
        output_dim = input_dim + 6
        h_conv_1 = conv_layer(h_from_prev, input_dim,
                              output_dim, "conv" + str(block + 1) + "_1")
        h_conv_2 = conv_layer(h_conv_1, output_dim,
                              output_dim, "conv" + str(block + 1) + "_2")
        h_pool = pooling_layer(h_conv_2, 'pool1' + str(block + 1))
        h_from_prev = h_pool

    # Reshape and fully connected
    with tf.name_scope('dropout'):
        h_pool10_flat = tf.reshape(h_from_prev, [-1, output_dim * 8 * 8 * 8])
        # To be able to turn it off during testing
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_pool_drop = tf.nn.dropout(h_pool10_flat, keep_prob)

    with tf.name_scope('fully_connected'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([output_dim * 8 * 8 * 8, 512])
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
            W_fc2 = weight_variable([512, num_cats])
            variable_summaries(W_fc2)
        with tf.name_scope('biases'):
            b_fc2 = bias_variable([num_cats])
            variable_summaries(b_fc2)
        with tf.name_scope('Wx_plus_b'):
            y_readout = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="y_readout")
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

    learning_rate = 0.001
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cross_entropy)

    cat_predicted = tf.argmax(y_readout, 1)
    cat_label = tf.argmax(y_, 1)
    with tf.name_scope('accuracy'):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(cat_predicted, cat_label)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("confusion_matrix"):
        num_classes = num_cats
        # Compute a per-batch confusion
        batch_confusion = tf.confusion_matrix(cat_label, cat_predicted,
                                              num_classes=num_classes,
                                              name='batch_confusion')

        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(tf.zeros([num_classes, num_classes],
                                         dtype=tf.int32),
                                name='confusion')
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = tf.assign(confusion, confusion + batch_confusion)

        confusion_image = tf.reshape(tf.cast(confusion_update, tf.float32),
                                     [1, num_classes, num_classes, 1])

        # Scale and colour
        current_max = tf.reduce_max(confusion_image, [0, 1, 2])

        # To [0, 1]
        confusion_image = confusion_image / current_max
        confusion_image_normalized = confusion_image / tf.reshape(tf.reduce_sum(confusion_image, [2]), [1, num_classes, 1, 1])

        min_color = tf.constant([255, 255, 255], dtype=tf.float32)
        max_color = tf.constant([49, 130, 189], dtype=tf.float32)
        color_vec = max_color - min_color

        confusion_image = confusion_image * color_vec + min_color
        confusion_image_normalized = confusion_image_normalized * color_vec + min_color

    tf.summary.image('confusion', confusion_image)
    tf.summary.image('confusion_normalized', confusion_image_normalized)

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(
        os.path.join(dir_name, "train"), sess.graph)
    test_writer = tf.summary.FileWriter(
        os.path.join(dir_name, "test"), sess.graph)

    # RUN THE TRAINING LOOPY LOOP
    tf.global_variables_initializer().run()
    total_steps = 0

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    for epoch in range(num_epochs):
        sess.run(training_iterator.initializer)
        step = -1
        print("Epoch: " + str(epoch))
        while True:
            step += 1
            if (step % 25 == 0):
                print(" - Step: " + str(step))
            try:
                if step % 10 == 0:  # Record summaries and test-set accuracy
                    sess.run(validation_iterator.initializer)
                    # Run the whole thing
                    summary, acc = sess.run([merged, accuracy], feed_dict={
                        handle: validation_handle, keep_prob: 1.0})
                    print('Accuracy at step %s: %s' % (step, acc))
                    test_writer.add_summary(
                        summary, total_steps + step)
                    save_path = saver.save(sess, os.path.join(
                        dir_name, "model.ckpt"))

                    print("Model saved in dir: %s" % dir_name)
                else:  # Record train set data summaries and train
                    if step % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([merged, train_step],
                                              feed_dict={handle: training_handle, keep_prob: 0.5})
                        train_writer.add_run_metadata(
                            run_metadata, 'epoch%10d step%10d' % (epoch, step))
                        train_writer.add_summary(
                            summary, total_steps + step)
                        save_path = saver.save(sess, os.path.join(
                            dir_name, "model.ckpt"))

                        print("Model saved in dir: %s" % dir_name)
                    else:  # Record a summary
                        summary, _, _ = sess.run([merged, train_step, confusion_update], feed_dict={
                            handle: training_handle, keep_prob: 0.5})

                        train_writer.add_summary(
                            summary, total_steps + step)
            except tf.errors.OutOfRangeError:
                total_steps += step
                break


def main():
    dir_name = sys.argv[1]
    res = int(sys.argv[2])
    train(dir_name, res)


if __name__ == "__main__":
    main()
