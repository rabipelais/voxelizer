This is a collection of scrips to voxelize OFF models into regular grids.
It downloads the ModelNet10 dataset.
To change certain parameters, such as the voxels resolution, or the number of processes, please edit the `create.py` file.

To run, type in `python ./create.py`

The resulting files will be in the folder `preprocessed-res-X` where X is the resolution (as in X^3)

To compile the C extension, simply run in the same folder:
`gcc -shared -o tribox.so -fPIC tribox.c`
Note that if you change the name of the library, you will also have to change the name (and location) on the python wrapper script.