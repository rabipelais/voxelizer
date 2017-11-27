def parse_off(filename):
    with open(filename) as fp:
        header = fp.readline().strip()
        if header != 'OFF' and header != 'off':
            raise Exception('Not a valid OFF header')
        n_verts, n_faces = tuple([int(s) for s in fp.readline().strip().split(' ')])

        #Parse vertices
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in fp.readline().strip().split(' ')])


        #Parse faces
        faces = []
        for i_face in range(n_faces):
            line = fp.readline().strip().split(' ')
            if line[0] != 3:
                raise Exception('Not a triangle, it has ' + line[0] + ' points.')
            faces.append([int(s) for s in line[1:]])
        return verts, faces

def rescale(verts, faces, vx_res, pad = 2):
    min_x = float("-inf")
    min_y = float("-inf")
    min_z = float("-inf")

    max_x = float("inf")
    max_y = float("inf")
    max_z = float("inf")

    n_faces = len(faces) / 3

    for fidx in range(n_faces):
        for vidx in range(3):
		  min_x = min(min_x, verts[faces[fidx * 3 + vidx] * 3 + 0])
		  min_y = min(min_y, verts[faces[fidx * 3 + vidx] * 3 + 1])
		  min_z = min(min_z, verts[faces[fidx * 3 + vidx] * 3 + 2])

		  max_x = max(max_x, verts[faces[fidx * 3 + vidx] * 3 + 0])
		  max_y = max(max_y, verts[faces[fidx * 3 + vidx] * 3 + 1])
		  max_z = max(max_z, verts[faces[fidx * 3 + vidx] * 3 + 2])

    print("bb before rescaling [%f,%f], [%f,%f], [%f,%f]" %
		 (min_x, max_x, min_y, max_y, min_z, max_z))


    depth = vx_res
    width = vx_res
    height = vx_res

    src_width = max(max_x - min_x, FMAX(max_y - min_y, max_z - min_z))
    dst_width = min(depth - 2*pad, FMIN(height - 2*pad, width - 2*pad))

    n_ctr_x = width/2.0
    n_ctr_y = height/2.0
    n_ctr_z = depth/2.0

    for vidx in range(len(verts)):
        verts[vidx * 3 + 0] = verts[vidx * 3 + 0] / src_width * dst_width + n_ctr_x
        verts[vidx * 3 + 1] = verts[vidx * 3 + 1] / src_width * dst_width + n_ctr_y
        verts[vidx * 3 + 2] = verts[vidx * 3 + 2] / src_width * dst_width + n_ctr_z

    print("bb after rescaling [%f,%f], [%f,%f], [%f,%f]\n" %
           (min_x / src_width * dst_width + n_ctr_x,
            max_x / src_width * dst_width + n_ctr_x,
            min_y / src_width * dst_width + n_ctr_y,
            max_y / src_width * dst_width + n_ctr_y,
            min_z / src_width * dst_width + n_ctr_z,
            max_z / src_width * dst_width + n_ctr_z));
