import sys
import numpy as np
import time
import os
from glob import glob
import random
import multiprocessing
import urllib
import zipfile
import argparse

from voxelizer import *

sys.path.append('.')

random.seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser(
    description='Create a voxelized grid of resolution RESOLUTION from FILEs. If no files are given, it will use the ModelNet10 dataset, and download it if not present in the folder `m10`.')

parser.add_argument('--resolution', '-r', type=int, required=True,
                    help='The resolution of the voxel grid.')

parser.add_argument('--processes', '-j', type=int, default=1,
                    help='The number of threads for the voxelizing (default: 1).')

parser.add_argument('source', nargs='*', metavar='FILE',
                    help='OFF files to voxelize.')

parser.add_argument('--destination', '-o',
                    help='Name of the output directory. If not given, will be `preprocessed-res-RESOLUTION`.')

args = parser.parse_args()

vx_res = args.resolution
pad = 2
if args.destination:
    out_root = args.destination
else:
    out_root = './preprocessed-res-' + str(vx_res)

n_rots = 1
n_processes = args.processes

# list all off files
off_paths = []

if len(args.source) < 1:
    # get MN10 data
    if not os.path.exists('mn10.zip'):
        print('downloading ModelNet10')
        mn10 = urllib.URLopener()
        mn10.retrieve(
            "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip", "mn10.zip")

    in_root = 'mn10'
    if not os.path.isdir(in_root):
        print('unzipping ModelNet10')
        mn10 = zipfile.ZipFile('mn10.zip', 'r')
        mn10.extractall(in_root)
        mn10.close()

    for root, dirs, files in os.walk(in_root):
        off_paths.extend(glob(os.path.join(root, '*.off')))
    off_paths.sort()
else:
    off_paths = args.source


# create out directory
if not os.path.isdir(out_root):
    os.makedirs(out_root)


# fix off header for MN meshes
print('fixing off headers')
for path in off_paths:
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    # parse header
    if lines[0].strip().lower() != 'off':
        print(path)
        print(lines[0])

        splits = lines[0][3:].strip().split(' ')
        n_verts = int(splits[0])
        n_faces = int(splits[1])
        n_other = int(splits[2])

        f = open(path, 'w')
        f.write('OFF\n')
        f.write('%d %d %d\n' % (n_verts, n_faces, n_other))
        for line in lines[1:]:
            f.write(line)
        f.close()

# create voxel grid from off mesh


def worker(rot_idx, rot, off_idx, off_path):
    print('%d/%d - %d/%d - %s' %
          (rot_idx + 1, n_rots, off_idx + 1, len(off_paths), off_path))

    phi = rot / 180.0 * np.pi
    R = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))
    rot_out_dir = out_root

    basename, ext = os.path.splitext(os.path.basename(off_path))
    train_test_prefix = os.path.basename(os.path.dirname(off_path))

    print('create voxels')
    t = time.time()
    grid = calculate_voxels_from_off(off_path, vx_res)
    print('  took %f[s]' % (time.time() - t))

    grid_out_path = os.path.join(
        rot_out_dir, '%s_%s.vox' % (train_test_prefix, basename))
    print('write bin - %s' % grid_out_path)
    t = time.time()
    write_grid(grid_out_path, grid, vx_res, vx_res, vx_res)
    print('  took %f[s]' % (time.time() - t))


start_t = time.time()
if n_processes > 1:
    pool = multiprocessing.Pool(processes=n_processes)

for rot_idx, rot in enumerate(np.linspace(0, 360, n_rots, endpoint=False)):
    rot_out_dir = os.path.join(out_root, 'rot%03d' % np.round(rot))
    if not os.path.isdir(rot_out_dir):
        os.makedirs(rot_out_dir)

    for off_idx, off_path in enumerate(off_paths):
        #print('%d pool.apply_async' % off_idx)
        if n_processes > 1:
            pool.apply_async(worker, args=(rot_idx, rot, off_idx, off_path,))
        else:
            worker(rot_idx, rot, off_idx, off_path)

if n_processes > 1:
    pool.close()
    pool.join()

print('create_data took %f[s]' % (time.time() - start_t))
