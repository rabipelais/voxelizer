#include "voxelgrid/core/core.h"
#include "voxelgrid/core/geometry.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

void block_triangles(float cx, float cy, float cz, float vd, float vh, float vw, float* verts, float* faces, std::vector<int>& tinds) {
    for(int fidx = 0; fidx < n_faces; ++fidx) {
		//voxel centre
		float3 vx_c;
		vx_c.x = cx;
		vx_c.y = cy;
		vx_c.z = cz;
		//voxel size
		float3 vx_w;
		vx_w.x = vw;
		vx_w.y = vh;
		vx_w.z = vd;

		float3 v0;
		v0.x = verts[faces[fidx * 3 + 0] * 3 + 0];
		v0.y = verts[faces[fidx * 3 + 0] * 3 + 1];
		v0.z = verts[faces[fidx * 3 + 0] * 3 + 2];
		float3 v1;
		v1.x = verts[faces[fidx * 3 + 1] * 3 + 0];
		v1.y = verts[faces[fidx * 3 + 1] * 3 + 1];
		v1.z = verts[faces[fidx * 3 + 1] * 3 + 2];
		float3 v2;
		v2.x = verts[faces[fidx * 3 + 2] * 3 + 0];
		v2.y = verts[faces[fidx * 3 + 2] * 3 + 1];
		v2.z = verts[faces[fidx * 3 + 2] * 3 + 2];

		bool tria_inter = intersection_triangle_voxel(vx_c, vx_w, v0, v1, v2);
		if(tria_inter) {
			tinds.push_back(1);
		} else {
			tinds.push_back(0);
		}
    }
}

voxelgrid* voxelgrid_create_from_off_cpu(const char* path, vg_size_t depth, vg_size_t height, vg_size_t width, const float R[9], bool fit, int fit_multiply, bool pack, int pad, int n_threads) {
  std::ifstream file(path);
  std::string line;
  std::stringstream ss;
  int line_nb = 0;

  printf("[INFO] parse off file\n");

  //parse header
  std::getline(file, line); ++line_nb;
  if(line != "off" && line != "OFF") {
    std::cout << "invalid header: " << line << std::endl;
    std::cout << path << std::endl;
    exit(-1);
  }

  //parse n vertices, n faces
  size_t n_verts, n_faces;
  std::getline(file, line); ++line_nb;
  ss << line;
  ss >> n_verts;
  ss >> n_faces;
  int dummy;
  ss >> dummy;

  //reserve memory for vertices and triangs
  std::vector<float> verts;
  std::vector<int> faces;

  //parse vertices
  float x,y,z;
  float x_,y_,z_;
  for(size_t idx = 0; idx < n_verts; ++idx) {
    std::getline(file, line); ++line_nb;
    ss.clear(); ss.str("");
    ss << line;
    ss >> x_;
    ss >> y_;
    ss >> z_;

    x = R[0] * x_ + R[1] * y_ + R[2] * z_;
    y = R[3] * x_ + R[4] * y_ + R[5] * z_;
    z = R[6] * x_ + R[7] * y_ + R[8] * z_;

    verts.push_back(x);
    verts.push_back(y);
    verts.push_back(z);
  }

  //parse faces
  for(size_t idx = 0; idx < n_faces; ++idx) {
    std::getline(file, line); ++line_nb;
    ss.clear(); ss.str("");
    ss << line;
    ss >> dummy;
    if(dummy != 3) {
      std::cout << "not a triangle, has " << dummy << " pts" << std::endl;
      exit(-1);
    }

    ss >> dummy; faces.push_back(dummy);
    ss >> dummy; faces.push_back(dummy);
    ss >> dummy; faces.push_back(dummy);
  }

  if(n_verts != verts.size() / 3) {
    std::cout << "n_verts in header differs from actual n_verts" << std::endl;
    exit(-1);
  }
  if(n_faces != faces.size() / 3) {
    std::cout << "n_faces in header differs from actual n_faces" << std::endl;
    exit(-1);
  }

  file.close();

  bool rescale = true;

  //Rescale the Bounding Box ======================================
  float min_x = 1e9;  float min_y = 1e9;  float min_z = 1e9;
  float max_x = -1e9; float max_y = -1e9; float max_z = -1e9;
  for(int fidx = 0; fidx < n_faces; ++fidx) {
      for(int vidx = 0; vidx < 3; ++vidx) {
		  min_x = FMIN(min_x, verts[faces[fidx * 3 + vidx] * 3 + 0]);
		  min_y = FMIN(min_y, verts[faces[fidx * 3 + vidx] * 3 + 1]);
		  min_z = FMIN(min_z, verts[faces[fidx * 3 + vidx] * 3 + 2]);

		  max_x = FMAX(max_x, verts[faces[fidx * 3 + vidx] * 3 + 0]);
		  max_y = FMAX(max_y, verts[faces[fidx * 3 + vidx] * 3 + 1]);
		  max_z = FMAX(max_z, verts[faces[fidx * 3 + vidx] * 3 + 2]);
      }
  }

  // rescale vertices
  printf("bb before rescaling [%f,%f], [%f,%f], [%f,%f]\n",
		 min_x, max_x, min_y, max_y, min_z, max_z);

  float src_width = FMAX(max_x - min_x, FMAX(max_y - min_y, max_z - min_z));
  float dst_width = FMIN(depth - 2*pad, FMIN(height - 2*pad, width - 2*pad));
  float o_ctr_x = (max_x + min_x)/2.f; float n_ctr_x = width/2.f;
  float o_ctr_y = (max_y + min_y)/2.f; float n_ctr_y = height/2.f;
  float o_ctr_z = (max_z + min_z)/2.f; float n_ctr_z = depth/2.f;
  for(int vidx = 0; vidx < n_verts; ++vidx) {
      verts[vidx * 3 + 0] = (verts[vidx * 3 + 0] - o_ctr_x) / src_width * dst_width + n_ctr_x;
      verts[vidx * 3 + 1] = (verts[vidx * 3 + 1] - o_ctr_y) / src_width * dst_width + n_ctr_y;
      verts[vidx * 3 + 2] = (verts[vidx * 3 + 2] - o_ctr_z) / src_width * dst_width + n_ctr_z;
  }

  printf("bb after rescaling [%f,%f], [%f,%f], [%f,%f]\n",
		 (min_x - o_ctr_x) / src_width * dst_width + n_ctr_x,
		 (max_x - o_ctr_x) / src_width * dst_width + n_ctr_x,
		 (min_y - o_ctr_y) / src_width * dst_width + n_ctr_y,
		 (max_y - o_ctr_y) / src_width * dst_width + n_ctr_y,
		 (min_z - o_ctr_z) / src_width * dst_width + n_ctr_z,
		 (max_z - o_ctr_z) / src_width * dst_width + n_ctr_z);

  printf("[INFO] create voxel grid from mesh\n");
  //-----------------------------------------------------------

  //Determine block triangle intersections ===========================
  int n_blocks = depth * height * width;
  std::vector<std::vector<int> > tinds;

  printf("  [VoxelGrid] determine block triangle intersections\n");
#if defined(_OPENMP)
  omp_set_num_threads(n_threads);
#endif
  #pragma omp parallel for
  for(int grid_idx = 0; grid_idx < n_blocks; ++grid_idx) {
      int gd = grid_idx / (grid_height * grid_width);
      int gh = (grid_idx / grid_width) % grid_height;
      int gw = grid_idx % grid_width;

      block_triangles(gw,gh,gd, 1,1,1, tinds[grid_idx]);
  }

  //octree* grid = octree_create_from_mesh_cpu(n_verts, &(verts[0]), n_faces, &(faces[0]), rescale, depth, height, width, fit, fit_multiply, pack, pad, n_threads);

  return grid;
}
