from ctypes import cdll, c_int, c_float

import numpy as np
import numpy.ctypeslib as npct

array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
lib = npct.load_library("libgrid", ".")

# setup the return types and argument types
lib.triBoxOverlap.restype = c_int
lib.triBoxOverlap.argtypes = [array_1d_float, array_1d_float, array_2d_float]


def tri_box_intersection(box_center, box_halfsize, triverts):
    return lib.triBoxOverlap(box_center, box_halfsize, triverts)


lib.blockTriangle.restype = c_int
lib.blockTriangle.argtypes = [c_float, c_float,
                              c_float, c_int, array_1d_float, c_int, array_1d_int]


def block_triangle_c(cx, cy, cz, n_verts, verts, n_faces, faces):
    return lib.blockTriangle(cx, cy, cz, n_verts, verts, n_faces, faces)


lib.calculateVoxels.argtypes = [
    c_int, array_1d_float, c_int, array_1d_int, c_int, c_int, c_int, array_1d_int]


def calculate_voxels_c(n_verts, verts, n_faces, faces, width, height, depth, out):
    return lib.calculateVoxels(n_verts, verts, n_faces, faces, width, height, depth, out)
