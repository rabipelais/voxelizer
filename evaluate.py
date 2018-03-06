import tensorflow as tf
import numpy as np
import math
import os
import sys
import argparse

def parseArgs():
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


def main():
    args = parseArgs()
    print args

if __name__ == "__main__":
    main()