import tensorflow as tf
import numpy as np
import math
import os
import sys
import argparse
from tensorflow.python.tools import inspect_checkpoint as chkp

import voxelizer

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', '-m',
                        help='Name of the model save file. Default: `./model.ckpt` (just the prefix, not the numbers or the postfix after the dash)')
    parser.add_argument('--labels', '-l',
                    help='File with the mappings to category names. Default: `./labels.txt`')
    parser.add_argument('source', metavar='FILE',
                    help='`.vox` input file.')

    args = parser.parse_args()

    source = args.source

    if args.model:
        model = args.model
    else:
        model = './model.ckpt'

    if args.labels:
        labels = args.labels
    else:
        labels = './labels.txt'

    res = {"model" : model,
           "labels": labels,
           "source": source}
    return res

def read_data(source):
    basename, ext = os.path.splitext(source)
    if ext != '.vox':
        print("Error: FILE needs to be a valid `.vox` file.")
        exit()
    grid = voxelizer.read_grid(source)

    return grid

def main():
    args = parse_args()
    print args

    grid = read_data(args['source'])

    sess = tf.Session()

    #Load graph metadata and restore weights
    saver = tf.train.import_meta_graph(args['model'] + '.meta')
    saver.restore(sess, args['model'])

    graph = tf.get_default_graph()

    y_readout = graph.get_tensor_by_name("readout/Wx_plus_b/y_readout:0")
    keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
    x_input = graph.get_tensor_by_name("x_input:0")

    result_vector = tf.nn.softmax(y_readout) * 100

    result = sess.run(result_vector, feed_dict={x_input: grid, keep_prob: 1.0})

    result = result[0]

    #Read labels
    with open(args["labels"]) as f:
        labels = f.readlines()

    #Remove trailing number
    labels = [x.strip().rsplit(' ', 1)[0] for x in labels]

    categories = zip(result, labels)

    for (r, cat) in categories:
        print("Category `{}`: {:3.2f}%".format(cat, r))

    maximal = max(categories, key=lambda item:item[0])[1]

    print("")
    print("I believe the correct prediction is: `{}`".format(maximal))

if __name__ == "__main__":
    main()
