import numpy as np

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
        verts[vidx * 3 + 0] = verts[vidx * 3 + 0] / src_width + 0.5
        verts[vidx * 3 + 1] = verts[vidx * 3 + 1] / src_width + 0.5
        verts[vidx * 3 + 2] = verts[vidx * 3 + 2] / src_width + 0.5

    print("bb after rescaling [%f,%f], [%f,%f], [%f,%f]\n" %
            (min_x / src_width + 0.5,
             max_x / src_width + 0.5,
             min_y / src_width + 0.5,
             max_y / src_width + 0.5,
             min_z / src_width + 0.5,
             max_z / src_width + 0.5))

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

      grid.append(block_triangles(gw / float(width) + (1.0/(2*float(width)))
                                  , gh / float(height) + (1.0/(2*float(height)))
                                  , gd / float(depth) + (1.0/(2*float(depth))), verts, faces))

    gridnp = np.array(grid)
    gridnp.reshape(width, height, depth)

    return gridnp

def block_triangles(cx, cy, cz, verts, faces):
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

        intersection = intersection_triangle_voxel(vx_c, v0, v1, v2)
        if intersection:
            return True
    return False #No face intersects

def intersection_triangle_voxel(vc, v0, v1, v2):

    # Based on Tomas Akenine-Moeller's work, for a more advanced solution, see Graphics Gems III, Triangle-Cube Intersection, pp. 236-239
    #    use separating axis theorem to test overlap between triangle and box
    #    need to test for overlap in these directions:
    #    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle
    #       we do not even need to test these)
    #    2) normal of the triangle
    #    3) crossproduct(edge from tri, {x,y,z}-directin)
    #       this gives 3x3=9 more tests

    #Center everything on (0, 0, 0)
    v0 = v0 - vc
    v1 = v1 - vc
    v2 = v2 - vc

    #Compute edges
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    #Compute point 3)
    fex = abs(e0[0])
    fey = abs(e0[1])
    fez = abs(e0[2])
    #X01
    p0 = e0[2]*v0[1] - e0[1]*v0[2]
    p2 = e0[2]*v2[1] - e0[1]*v2[2]
    if(p0<p2):
        min_i = p0
        max_i = p2
    else:
        min_i = p2
        max_i = p0
    rad = fez * 0.5 + fey * 0.5
    if (min_i > rad or max_i < -rad):
        return False
    #Y02
    p0 = -e0[2]*v0[0] + e0[0]*v0[2]
    p2 = -e0[2]*v2[0] + e0[0]*v2[2]
    if(p0<p2):
        min_i = p0
        max_i = p2
    else:
        min_i = p2
        max_i = p0
    rad = fez * 0.5 + fex * 0.5
    if(min_i>rad or max_i<-rad):
        return False
    #Z12
    p1 = e0[1]*v1[0] - e0[0]*v1[1]
    p2 = e0[1]*v2[0] - e0[0]*v2[1]
    if(p2<p1):
        min_i = p2
        max_i = p1
    else:
        min_i = p1
        max_i = p2
    rad = fey * 0.5 + fex * 0.5
    if(min_i >rad or max_i<-rad):
        return False


    fex = abs(e1[0])
    fey = abs(e1[1])
    fez = abs(e1[2])
    #X01
    p0 = e1[2]*v0[1] - e1[1]*v0[2]
    p2 = e1[2]*v2[1] - e1[1]*v2[2]
    if(p0<p2):
        min_i = p0
        max_i = p2
    else:
        min_i = p2
        max_i = p0
    rad = fez * 0.5 + fey * 0.5
    if (min_i > rad or max_i < -rad):
        return False
    #Y02
    p0 = -e1[2]*v0[0] + e1[0]*v0[2]
    p2 = -e1[2]*v2[0] + e1[0]*v2[2]
    if(p0<p2):
        min_i = p0
        max_i = p2
    else:
        min_i = p2
        max_i = p0
    rad = fez * 0.5 + fex * 0.5
    if(min_i>rad or max_i<-rad):
        return False
    #Z0
    p0 = e1[1]*v0[0] - e1[0]*v0[1]
    p1 = e1[1]*v1[0] - e1[0]*v1[1]
    if(p0<p1):
        min_i = p0
        max_i = p1
    else:
        min_i = p1
        max_i = p0
	rad = fey * 0.5 + fex * 0.5
    if(min_i > rad or max_i < -rad):
        return False


    fex = abs(e2[0])
    fey = abs(e2[1])
    fez = abs(e2[2])
    #X2
    p0 = e2[2]*v0[1] - e2[1]*v0[2]
    p1 = e2[2]*v1[1] - e2[1]*v1[2]
    if(p0<p1):
        min_i = p0
        max_i = p1
    else:
        min_i = p1
        max_i = p0
    rad = fez * 0.5 + fey * 0.5
    if (min_i > rad or max_i < -rad):
        return False
    #Y1
    p0 = -e2[2]*v0[0] + e2[0]*v0[2]
    p2 = -e2[2]*v1[0] + e2[0]*v1[2]
    if(p0<p2):
        min_i = p0
        max_i = p2
    else:
        min_i = p2
        max_i = p0
    rad = fez * 0.5 + fex * 0.5
    if(min_i>rad or max_i<-rad):
        return False
    #Z12
    p1 = e2[1]*v1[0] - e2[0]*v1[1]
    p2 = e2[1]*v2[0] - e2[0]*v2[1]
    if(p2<p1):
        min_i = p2
        max_i = p1
    else:
        min_i = p1
        max_i = p2
    rad = fey * 0.5 + fex * 0.5
    if(min_i>rad or max_i<-rad):
        return False


    # Bullet point 1)
    #  first test overlap in the {x,y,z}-directions
    #  find min, max of the triangle each direction, and test for overlap in
    #  that direction -- this is equivalent to testing a minimal AABB around
    #  the triangle against the AABB
    min_0 = min([v0[0], v1[0], v2[0]])
    max_0 = max([v0[0], v1[0], v2[0]])
    if(min_0 > 0.5 or max_0 < -0.5):
        return False

    min_1 = min([v0[1], v1[1], v2[1]])
    max_1 = max([v0[1], v1[1], v2[1]])
    if(min_1 > 0.5 or max_1 < -0.5):
        return False

    min_2 = min([v0[2], v1[2], v2[2]])
    max_2 = max([v0[2], v1[2], v2[2]])
    if(min_2 > 0.5 or max_2 < -0.5):
        return False

    # Bullet 2)
    # test if box intersects plane of the triangle
    normal = np.cross(e0, e1)

    vmin = np.sign(normal) * (-0.5) - v0
    vmax = np.sign(normal) * -0.5 - v0

    intersect_normal = False

    if np.dot(normal, vmin) > 0.0:
        intersect_normal = False
    elif np.dot(normal, vmax) >= 0.0:
        intersect_normal = True
    else:
        intersect_normal = False

    if not intersect_normal:
        return False

    return True #Box and triangle overlap


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
