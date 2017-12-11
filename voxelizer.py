import numpy as np
import tribox_wrapper

def parse_off(filename):
    with open(filename) as fp:
        header = fp.readline().strip()
        if header != 'OFF' and header != 'off':
            raise Exception('Not a valid OFF header')
        n_verts, n_faces, _ = tuple([int(s) for s in fp.readline().strip().split(' ')])

        #Parse vertices
        verts = []
        for i_vert in range(n_verts):
            verts.extend([float(s) for s in fp.readline().strip().split(' ')])

        #Parse faces
        faces = []
        for i_face in range(n_faces):
            line = fp.readline().strip().split(' ')
            if int(line[0]) != 3:
                raise Exception('Not a triangle, it has ' + line[0] + ' points.')
            faces.extend([int(s) for s in line[1:]])
        return verts, faces

def rescale(verts, faces, vx_res, pad = 2):
    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")

    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")

    n_faces = len(faces) / 3

    for fidx in range(n_faces / 3):
        for vidx in range(3):
		  min_x = min(min_x, verts[faces[fidx * 3 + vidx] * 3 + 0])
		  min_y = min(min_y, verts[faces[fidx * 3 + vidx] * 3 + 1])
		  min_z = min(min_z, verts[faces[fidx * 3 + vidx] * 3 + 2])

		  max_x = max(max_x, verts[faces[fidx * 3 + vidx] * 3 + 0])
		  max_y = max(max_y, verts[faces[fidx * 3 + vidx] * 3 + 1])
		  max_z = max(max_z, verts[faces[fidx * 3 + vidx] * 3 + 2])

    #print("bb before rescaling [%f,%f], [%f,%f], [%f,%f]" %
#		 (min_x, max_x, min_y, max_y, min_z, max_z))


    depth = vx_res
    width = vx_res
    height = vx_res

    src_width = max(max_x - min_x, max(max_y - min_y, max_z - min_z))
    dst_width = min(depth, min(height, width))

    n_ctr_x = width/2.0
    n_ctr_y = height/2.0
    n_ctr_z = depth/2.0

    for vidx in range(len(verts) / 3):
        verts[vidx * 3 + 0] = dst_width * (verts[vidx * 3 + 0] / src_width + 0.5)
        verts[vidx * 3 + 1] = dst_width * (verts[vidx * 3 + 1] / src_width + 0.5)
        verts[vidx * 3 + 2] = dst_width * (verts[vidx * 3 + 2] / src_width + 0.5)

    print("bb after rescaling [%f,%f], [%f,%f], [%f,%f]\n" %
            (dst_width * (min_x / src_width + 0.5),
             dst_width * (max_x / src_width + 0.5),
             dst_width * (min_y / src_width + 0.5),
             dst_width * (max_y / src_width + 0.5),
             dst_width * (min_z / src_width + 0.5),
             dst_width * (max_z / src_width + 0.5)))

    return verts, faces


def calculate_voxels(verts, faces, width, height, depth):
    n_blocks = width * height * depth

    grid = []
    #dumb for loop, parallelize, in order whd
    #should only parallel with preallocation to not mess with ordering
    for grid_idx in range(n_blocks):
      gd = grid_idx // (height * width) # // forces integer division
      gh = (grid_idx // width) % height
      gw = grid_idx % width

      # grid.append(block_triangles(gw / float(width) + (1.0/(2*float(width)))
      #                             , gh / float(height) + (1.0/(2*float(height)))
      #                             , gd / float(depth) + (1.0/(2*float(depth))), verts, faces))

      grid.append(block_triangles(gw + 0.5, gh + 0.5, gd + 0.5, verts, faces))

    gridnp = np.array(grid)
    gridnp.reshape(width, height, depth)

    return gridnp

def block_triangles(cx, cy, cz, verts, faces):
    half_sizes = np.array([0.5, 0.5, 0.5])
    for fidx in range(len(faces) / 3):
        vx_c = np.array([cx, cy, cz])

        v0 = np.array([
              verts[faces[fidx * 3 + 0] * 3 + 0]
            , verts[faces[fidx * 3 + 0] * 3 + 1]
            , verts[faces[fidx * 3 + 0] * 3 + 2]])

        v1 = np.array([
              verts[faces[fidx * 3 + 1] * 3 + 0]
            , verts[faces[fidx * 3 + 1] * 3 + 1]
            , verts[faces[fidx * 3 + 1] * 3 + 2]])
        v2 = np.array([
              verts[faces[fidx * 3 + 2] * 3 + 0]
            , verts[faces[fidx * 3 + 2] * 3 + 1]
            , verts[faces[fidx * 3 + 2] * 3 + 2]])

        verts_array = np.array([v0, v1, v2])
        intersection = tribox_wrapper.tri_box_intersection(np.float32(vx_c)
                                                           , np.float32(half_sizes)
                                                           , np.float32(verts_array))
        if intersection == 1:
            return True
    return False #No face intersects

# Filename and resolution. Assume cube?
def calculate_voxels_from_off(filename, res):
    verts, faces = parse_off(filename)

    #verts, faces = rotate(verts, faces)

    verts, faces = rescale(verts, faces, res)

    grid_array = calculate_voxels(verts, faces, res, res, res)

    return map(lambda x: 1 if x else 0, grid_array)

def write_grid(filename, grid, width, height, depth):
    with open(filename, 'w') as fp:
        fp.write(str(width) + ' ' + str(height) + ' ' + str(depth) + '\n')
        fp.write(' '.join(map(str, np.ravel(grid))))

def read_grid(filename):
    with open(filename) as fp:
        width, height, depth = tuple([int(s) for s in fp.readline().strip().split(' ')])
        array = fp.readline().strip()

        grid = np.array(array)
        grid.reshape(width, height, depth)

        return grid
