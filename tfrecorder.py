import tensorflow as tf
import numpy as np
import os
import sys

import voxelizer


def read_data(dir):
    labels_dict = {}
    current_label_idx = 0
    # Read the data
    for (dirpath, dirnames, filenames) in os.walk(dir):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for name in filenames:
            basename, ext = os.path.splitext(name)
            if ext != '.vox':
                continue
            grid = voxelizer.read_grid(os.path.join(dirpath, name))

            components = basename.split('_')
            train_p = components[0]
            label_name = "-".join(components[1:-1])

            label_idx = -1
            if label_name in labels_dict:
                label_idx = labels_dict[label_name]
            else:
                labels_dict[label_name] = current_label_idx
                label_idx = current_label_idx
                current_label_idx += 1

            if train_p == "train":
                train_data.append(grid)
                train_labels.append(label_idx)
            else:
                test_data.append(grid)
                test_labels.append(label_idx)

        train = zip(train_data, one_hot(train_labels, len(labels_dict)))
        test = zip(test_data, one_hot(test_labels, len(labels_dict)))

        return train, test, labels_dict


def one_hot(indices, cats):
    one_hots = []
    for i in indices:
        hotty = np.zeros(cats, dtype=np.int)
        hotty[i] = 1
        one_hots.append(hotty)
    return np.array(one_hots)


def _floats_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    dir_name = sys.argv[1]
    train_set, test_set, labels_dict = read_data(dir_name)

    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(dir_name, "training.tfrecord"))
    for data, label in test_set:
        (width, height, depth) = data.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'width': _int64_feature(width),
            'height': _int64_feature(height),
            'depth': _int64_feature(depth),
            'data': _floats_feature(data),
            'label_one_hot': _int64_feature()}))
