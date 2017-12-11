from ctypes import cdll, c_int

import numpy as np
import numpy.ctypeslib as npct

array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
lib = npct.load_library("tribox", ".")

# setup the return types and argument types
lib.triBoxOverlap.restype = c_int
lib.triBoxOverlap.argtypes = [array_1d_float, array_1d_float, array_2d_float]

def tri_box_intersection(box_center, box_halfsize, triverts):
    return lib.triBoxOverlap(box_center, box_halfsize, triverts)
