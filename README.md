# Turbo Mega Voxelizer 3000

This is a collection of scripts to voxelize OFF models into regular voxel grids.

## Creating the data
Use the file `create.py`, e.g.:

        python create.py --resolution 128 file1.off file2.off

If no files list is given, it will search for the ModelNet10 files in `m10`, and otherwise download ModelNet10
For further options, such as output directory and number of threads, run `python create.py --help`.

By default, the files will be stored in the directory `preprocessed-res-$RESOLUTION`. Each file is a 1D np.array of `1`s for voxels and `0`s for empty space in WHD order.

To compile the C extension, simply run in the same folder:
`gcc -shared -o grid.so -fPIC tribox.c`
Note that if you change the name of the library, you will also have to change the name (and location) on the python wrapper script.
