# Turbo Mega Voxelizer 3000

This is a collection of scripts to voxelize OFF models into regular voxel grids.

## Creating the data
To be able to run the voxelizer, first compile the C extension. Simply run in the same folder:

        gcc -shared -o libgrid.so -fPIC -fopenmp tribox.c

Note that if you change the name of the library, you will also have to change the name (and location) on the python wrapper script.

Now we can use the file `create.py`, e.g.:

        python create.py --resolution 128 file1.off file2.off

If no files list is given, it will search for the ModelNet10 files in `m10`, and otherwise download ModelNet10
For further options, such as output directory and number of threads, run `python create.py --help`.

By default, the files will be stored in the directory `preprocessed-res-$RESOLUTION`. Each file is a 1D np.array of `1`s for voxels and `0`s for empty space in WHD order.

## Formatting the data for TensorFlow™
To make importing the data easier into TensorFlow™, we should merge the files into a `TFRecord`. The command `python tfrecorder.py preprocessed-res-32` will merge the `.vox` files in the directory into one `training.tfrecord`, one `test.tfrecord`, and a `labels.txt` label to id correspondence file. The categories in the records are encoded as one-hot vectors.

It assumes the input files have the following format: `{test,train}_LABEL_number.vox`, where `LABEL` is the category of the object, and number an arbitrary identifier. It will output one file for the test data, and one for the training data, and a text file with a label-class id correspondence.
Example file name: `test_bathtub_0229.vox`

Run with the `--help` option for more options.


## Training and evaluating the model
Simply run `python train.py DIR RESOLUTION`, where `DIR` is the data directory with the TFRecords and `RESOLUTION` is the voxel grid resolution. It will train the model and evaluate it with the test data and write tensorboard files into `DIR/train/` and `DIR/test` respectively.

To view the TensorBoard results, run

        tensorboard --logdir DIR --port 6006

and you can now view it on Internet Explorer 6 at `localhost:6006`.


## Using the trained model for prediction
To use a model you trained in the last step to predict the category of a voxelized object, run the following command: `python evaluate.py OBJECT.vox -m /DIR/TO/MODEL/model.ckpt -l /DIR/TO/LABELS/labels.txt`, for example:

`python evaluate.py ~/Documents/TrainingSet4-OnlyWashers/train_FLAT\ WASHER_001.vox  -m ~/Documents/TrainingSet4-OnlyWashers/model.ckpt -l ~/Documents/TrainingSet4-OnlyWashers/labels.txt `

Run with the `--help` option for more information.
