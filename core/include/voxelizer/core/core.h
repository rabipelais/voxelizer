#ifndef OCTREE_H
#define OCTREE_H

#ifndef DEBUG
#define DEBUG 0
#endif

#include <string>
#include <cstdio>
#include <cmath>

#include <smmintrin.h>

/// data type used for to indicate the shape
typedef int vg_size_t;

/// data type to encode the shallow octree data structure as bit string
typedef int vg_tree_t;

/// data type for data arrays
typedef float vg_data_t;

typedef struct {
  vg_size_t n;             ///< number of grid-octrees (batch size).
  vg_size_t grid_depth;    ///< number of shallow octrees in the depth dimension.
  vg_size_t grid_height;   ///< number of shallow octrees in the height dimension.
  vg_size_t grid_width;    ///< number of shallow octrees in the width dimension.

  vg_size_t feature_size;  ///< length of the data vector associated with a single cell.
  vg_data_t* data;         ///< contiguous data array, all feature vectors associated with the grid-octree data structure.
} voxelgrid;


voxelgrid* voxelgrid_create_from_off(const char* path, vg_size_t depth, vg_size_t height, vg_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads);



#endif
