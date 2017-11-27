cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()

"""
Lightweight wrapper class for native float arrays. Allows numpy style access.
"""
cdef class FloatArrayWrapper:
  """ native float array that is encapsulated. """
  cdef float* data_ptr
  """ size/length of the float array. """
  cdef int size
  """ indicates owner ship of the float array, if true, free array in destructor. """
  cdef int owns

  """
  Set the native data array for this class.
  @param data_ptr native float array
  @param size length of the array
  @param owns if True, the object destructor frees the array
  """
  cdef set_data(self, float* data_ptr, int size, int owns):
    self.data_ptr = data_ptr
    self.size = size
    self.owns = owns

  """
  Method, which is called by np.array to encapsulate the data provided in data_ptr.
  """
  def __array__(self):
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> self.size
    ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT, self.data_ptr)
    return ndarray

  """
  Destructor. Calls free on data array if it is owned (owns == True) by the wrapper.
  """
  def __dealloc__(self):
    cdef float* ptr = self.data_ptr
    if self.owns != 0:
      free(<void*>ptr)


cdef extern from "../core/include/voxelizer/core/core.h":
  ctypedef int vg_size_t;
  ctypedef float vg_data_t;
  ctypedef int vg_tree_t;
  ctypedef struct voxelgrid:
    ot_size_t n;
    ot_size_t grid_depth;
    ot_size_t grid_height;
    ot_size_t grid_width;
    ot_size_t feature_size;
    ot_data_t* data;
  voxelgrid* voxelgrid_create_from_off(const char* path, vg_size_t depth, vg_size_t height, vg_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);


"""
Simple helper function that converts a python list of integers representing the
shape of an array to an native int array.
@param shape list of int.
@return native int array.
"""
cdef int* npy_shape_to_int_array(shape):
  cdef int* dims = <int*> malloc(len(shape) * sizeof(int))
  for idx in range(len(shape)):
    dims[idx] = shape[idx]
  return dims

"""
Reads a dense tensor from a binary file to a preallocated array.
@param path path to binary file.
@param data 5-dimensional, contiguous data array.
"""
def read_dense(char* path, float[:,:,:,:,::1] data):
  cdef int* dims = npy_shape_to_int_array(data.shape)
  dense_read_prealloc_cpu(path, 5, dims, &(data[0,0,0,0,0]))
  free(dims)

"""
Writes a dense tensor to a binary file.
@param path output path.
@param data data tensor.
"""
def write_dense(char* path, float[:,:,:,:,::1] data):
  cdef int* dims = npy_shape_to_int_array(data.shape)
  dense_write_cpu(path, 5, dims, &(data[0,0,0,0,0]))
  free(dims)


"""

  # To create an empty voxelgrid call
  VoxelGrid.create_empty()

  # To load a serialized voxelgrid call
  VoxelGrid.create_from_bin('/path/to/voxelgrid.oc')
"""
cdef class VoxelGrid:
  """ Pointer to native hybrid grid-voxelgrid structure. """
  cdef voxelgrid* grid

  """
  Set the pointer to native voxelgrid wrapped by this object.
  @note the voxelgrid_free_cpu is called on destruction of this wrapper, therefore
        freeing all the memory associated with the native voxelgrid.
  @param grid pointer to native voxelgrid structure.
  """
  cdef set_grid(self, voxelgrid* grid):
    self.grid = grid

  """
  Get pointer to native voxelgrid structure.
  @return voxelgrid*
  """
  cdef voxelgrid* get_grid(self):
    return self.grid


  """
  Destructor. Calls free on the wrapped voxelgrid structure.
  """
  def __dealloc__(self):
    voxelgrid_free_cpu(self.grid)

  """
  Prints the voxelgrid data structure to stdout.
  """
  def print_tree(self):
    voxelgrid_print_cpu(self.grid)

  """ @return the batch size of the voxelgrid structure. """
  def n(self):
    return self.grid[0].n
  """ @return the grid depth of the voxelgrid structure. """
  def grid_depth(self):
    return self.grid[0].grid_depth
  """ @return the grid height of the voxelgrid structure. """
  def grid_height(self):
    return self.grid[0].grid_height
  """ @return the grid width of the voxelgrid structure. """
  def grid_width(self):
    return self.grid[0].grid_width
  """ @return the feature size of the voxelgrid structure. """
  def feature_size(self):
    return self.grid[0].feature_size
  """ @return the depth of the tensor that corresponds to the voxelgrid structure. """
  def vx_depth(self):
    return self.grid[0].grid_depth
  """ @return the height of the tensor that corresponds to the voxelgrid structure. """
  def vx_height(self):
    return self.grid[0].grid_height
  """ @return the width of the tensor that corresponds to the voxelgrid structure. """
  def vx_width(self):
    return self.grid[0].grid_width

  """
  Returns the number of bytes that are reserved for this voxelgrid instance.
  This is a upper bound of what is really needed by this instance.
  @return number ob bytes allocated for this voxelgrid instance.
  """
  def mem_capacity(self):
    return voxelgrid_mem_capacity(self.grid)

  """
  Returns the number of bytes that are necessary for this voxelgrid instance.
  @return number of bytes needed for this voxelgrid instance.
  """
  def mem_using(self):
    return voxelgrid_mem_using(self.grid)

  """
  Compares this voxelgrid with another VoxelGrid (shape, structure, data).
  @return True, if VoxelGrid other is equal, False otherwise.
  """
  def equals(self, VoxelGrid other):
    return voxelgrid_equal_cpu(self.grid, other.grid)

  """
  Class method that creates and wraps an empty native voxelgrid.
  @return VoxelGrid wrapper.
  """
  @classmethod
  def create_empty(cls):
    cdef voxelgrid* ret = voxelgrid_new_cpu()
    cdef VoxelGrid grid = VoxelGrid()
    grid.set_grid(ret)
    return grid

  """
  Class method that reads a native voxelgrid from the given path.
  @param path
  @return VoxelGrid wrapper.
  """
  @classmethod
  def create_from_bin(cls, char* path):
    cdef voxelgrid* ret = voxelgrid_new_cpu()
    voxelgrid_read_cpu(path, ret)
    cdef VoxelGrid grid = VoxelGrid()
    grid.set_grid(ret)
    return grid

  """
  Creates a native clone of the voxelgrid and wraps it in a new Python object.
  @return VoxelGrid wrapper.
  """
  def copy(self):
    cdef VoxelGrid other = self.create_empty()
    voxelgrid_copy_cpu(self.grid, other.grid)
    return other

  """
  Converts the voxelgrid to a tensor where the features are the last dimension.
  n x depth x height x width x features.
  @return numpy array.
  """
  def to_dhwc(self):
    dense = np.empty((self.n(), self.vx_depth(), self.vx_height(), self.vx_width(), self.feature_size()), dtype=np.float32)
    cdef float[:,:,:,:,::1] dense_view = dense
    voxelgrid_to_dhwc_cpu(self.grid, self.vx_depth(), self.vx_height(), self.vx_width(), &(dense_view[0,0,0,0,0]))
    return np.squeeze(dense)

  """
  Converts the voxelgrid to a tensor where the features are the second dimension.
  n x features x depth x height x width.
  @return numpy array.
  """
  def to_cdhw(self):
    dense = np.empty((self.n(), self.feature_size(), self.vx_depth(), self.vx_height(), self.vx_width()), dtype=np.float32)
    cdef float[:,:,:,:,::1] dense_view = dense
    voxelgrid_to_cdhw_cpu(self.grid, self.vx_depth(), self.vx_height(), self.vx_width(), &(dense_view[0,0,0,0,0]))
    return np.squeeze(dense)

  """
  Serializes the voxelgrid to a binary file.
  @param path
  """
  def write_bin(self, char* path):
    voxelgrid_write_cpu(path, self.grid)

  """
  First converts the voxelgrid to a tensor and then serializes the tensor to a
  binary file.
  @param path
  """
  def write_to_cdhw(self, char* path):
    voxelgrid_cdhw_write_cpu(path, self.grid)

  """
  Class method to create an voxelgrid structure from an OFF file (mesh).
  @param path
  @param depth number of voxel in depth dimension the voxelgrid should comprise.
  @param height number of voxel in height dimension the voxelgrid should comprise.
  @param width number of voxel in width dimension the voxelgrid should comprise.
  @param R 3x3 rotation matrix applied to the triangle mesh
  @param fit deprecated
  @param fit_multiply deprecated
  @param pack deprecated
  @param n_threads number of CPU threads that should be used for this function.
  """
  @classmethod
  def create_from_off(cls, char* path, ot_size_t depth, ot_size_t height, ot_size_t width, float[:,::1] R, bool fit=False, int fit_multiply=1, bool pack=False, int pad=0, int n_threads=1):
    if R.shape[0] != 3 or R.shape[1] != 3:
      raise Exception('invalid R shape')
    cdef voxelgrid* ret = voxelgrid_create_from_off_cpu(path, depth, height, width, &(R[0,0]), fit, fit_multiply, pack, pad, n_threads)
    cdef VoxelGrid grid = VoxelGrid()
    grid.set_grid(ret)
    return grid


"""
Warps a native voxelgrid struct in a Python VoxelGrid class.
@return VoxelGrid wrapper.
"""
cdef warp_voxelgrid(voxelgrid* grid):
  grid_w = VoxelGrid()
  grid_w.set_grid(grid)
  return grid_w
