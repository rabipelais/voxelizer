import os
import sys
import argparse
from shutil import copyfile


def parse_obj_file(f, processed):
    path = os.path.normpath(f).split(os.path.sep)
    cat = path[0]
    train_test = path[1]
    basename = path[2]

    components = basename.split('_')
    number = components[0]
    rest = components[1]
    label_parts = rest.split('.')
    label = label_parts[0]
    label = label.replace(" ", "-")
    label = label.lower()

    new_name = "_".join([train_test, label, number]) + '.vox'

    isfile = os.path.isfile(os.path.join(processed, new_name))

    if not isfile:
        print("FAILED TO FIND: " + f)
        print("TRIED: " + new_name)
        return ()

    dest = os.path.join(cat, train_test)
    copyfile(os.path.join(processed, new_name), os.path.join(dest, new_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument('--processed', '-p')
    parser.add_argument('source', nargs='*', metavar='FILE')
    args = parser.parse_args()

    files = args.source
    processed = args.processed

    for f in files:
        parse_obj_file(f, processed)
