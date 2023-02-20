# import open3d as o3d
import os
import numpy as np
import open3d as o3d

# input_dir = "/root/studio/project/shine-mapping/SHINE_mapping/experiments/ncd/" #文件夹目录
# input_dir = "/root/studio/project/shine-mapping/SHINE_mapping/experiments/kitti/07/" #文件夹目录
# input_dir = "/root/studio/project/rundc/rundc/samples/" #文件夹目录
input_dir = "/root/studio/project/rundc/rundc/samples/0203/" #文件夹目录 ncd ours



# dir_list = sorted(os.listdir(input_dir))
# print(dir_list)



# first_mesh_path = "./samples/gt_100_scans_07_kitti-odometry_range_30_start_0" + "_reorient_mesh.ply"
first_mesh_path = input_dir + "gt_100_scans_01_quad_range_30_start_0" + "_reorient_mesh.ply" # ours ncd

# first_mesh_path = input_dir + dir_list[0] + "/mesh/mesh_iter_20000.ply"

mesh = o3d.io.read_triangle_mesh(first_mesh_path)

# output_mesh_dir = input_dir + "gt_100_scans_07_kitti-odometry_range_30_reorient_mesh.ply"
output_mesh_dir = input_dir + "gt_100_scans_01_quad_range_30_reorient_mesh.ply" # ours ncd


for ii in range(1, 11):
    tmp_mesh_path = input_dir + "gt_100_scans_01_quad_range_30_start_" + str(ii*100) + "_reorient_mesh.ply"
    # tmp_mesh_path = input_dir + dir_list[ii] + "/mesh/mesh_iter_20000.ply"
    
    tmp_mesh = o3d.io.read_triangle_mesh(tmp_mesh_path)
    mesh += tmp_mesh

o3d.io.write_triangle_mesh(output_mesh_dir, mesh)



    
    