import tensorflow as tf
import numpy as np
import os

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

train_data_dir = "preprocessed"

labels_dict = {}
current_label_idx = 0
vx_width = 64
vx_height = 64
vx_depth = 64

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def build_graph(x):
    with tf.name_scope('reshape'):
        x_voxel = tf.reshape(x, [-1, vx_width, vx_height, vx_depth, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 3, 1, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
      h_pool1 = tf.nn.max_pool3d(h_conv1, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 3, 8, 14])
        b_conv2 = bias_variable([14])
        h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
      h_pool2 = tf.nn.max_pool3d(h_conv2, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 3, 14, 14])
        b_conv3 = bias_variable([14])
        h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv2) + b_conv2)

    with tf.name_scope('pool3'):
      h_pool3 = tf.nn.max_pool3d(h_conv3, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 3, 14, 20])
        b_conv4 = bias_variable([20])
        h_conv4 = tf.nn.relu(conv3d(h_pool3, W_conv2) + b_conv3)

    with tf.name_scope('pool4'):
      h_pool4 = tf.nn.max_pool3d(h_conv4, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 3, 20, 20])
        b_conv5 = bias_variable([20])
        h_conv5 = tf.nn.relu(conv3d(h_pool4, W_conv2) + b_conv4)

    with tf.name_scope('pool5'):
      h_pool5 = tf.nn.max_pool3d(h_conv5, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 3, 20, 26])
        b_conv6 = bias_variable([26])
        h_conv6 = tf.nn.relu(conv3d(h_pool5, W_conv2) + b_conv5)

    with tf.name_scope('pool6'):
      h_pool6 = tf.nn.max_pool3d(h_conv6, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')


    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([3, 3, 3, 26, 26])
        b_conv7 = bias_variable([26])
        h_conv7 = tf.nn.relu(conv3d(h_conv6, W_conv6) + b_conv6)
    with tf.name_scope('conv8'):
        W_conv8 = weight_variable([3, 3, 3, 26, 32])
        b_conv8 = bias_variable([32])
        h_conv8 = tf.nn.relu(conv3d(h_conv7, W_conv7) + b_conv7)
    with tf.name_scope('conv9'):
        W_conv9 = weight_variable([3, 3, 3, 32, 32])
        b_conv9 = bias_variable([32])
        h_conv9 = tf.nn.relu(conv3d(h_conv8, W_conv8) + b_conv8)
    with tf.name_scope('conv10'):
        W_conv10 = weight_variable([3, 3, 3, 32, 32])
        b_conv10 = bias_variable([32])
        h_conv10 = tf.nn.relu(conv3d(h_conv9, W_conv9) + b_conv9)

    with tf.name_scope('pool10'):
      h_pool10 = tf.nn.max_pool3d(h_conv10, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')

    #Reshape and fully connected
    with tf.name_scope('dropout'):
        h_pool10_flat = tf.reshape(-1, 32 * 8 * 8 * 8)
        keep_prob = tf.placeholder(tf.float32)
        h_pool_drop = tf.nn.dropout(h_pool10_flat, keep_prob)

        W_fc1 = weight_variable([32 * 8 * 8 * 8, 512])
        b_fc1 = bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_drop, W_fc1) + b_fc1)


        W_fc2 = weight_variable([512, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def read_data():
  # Read the data
  for (dirpath, dirnames, filenames) in os.walk(train_data_dir):
    data_list = []
    labels_list = []
    for name in files:
      basename, ext = os.path.spliteext(name)
      if ext != 'vox':
        break
      grid = read_grid(os.path.join(dirpath, name))
      data.append(grid)
      train, label_name, number = tuple(basename.split('_'))

      label_idx = -1
      if label_name in labels_dict:
        label_idx = labels_dict[label_name]
      else:
        labels_dict[label_name] = current_label_idx
        label_idx = current_label_idx
        curren_label_idx += 1

        labels_list.append(labels_idx)

    labels_one_hot = tf.one_hot(labels_list, len(labels_dict))

    return data_list, labels_one_hot


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


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50) #!!!!!!
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
