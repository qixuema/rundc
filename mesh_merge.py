# import open3d as o3d
import os
import numpy as np
import open3d as o3d

first_mesh_path = "./samples/gt_100_scans_06_kitti-odometry_range_30_start_0" + "_reorient_mesh.ply"
mesh = o3d.io.read_triangle_mesh(first_mesh_path)


input_dir = "./samples/" #文件夹目录
output_mesh_dir = "./samples/gt_100_scans_06_kitti-odometry_range_30_reorient_mesh.ply"

for ii in range(1, 11):
    tmp_mesh_path = input_dir + "gt_100_scans_06_kitti-odometry_range_30_start_" + str(ii*100) + "_reorient_mesh.ply"
    tmp_mesh = o3d.io.read_triangle_mesh(tmp_mesh_path)
    mesh += tmp_mesh
    
o3d.io.write_triangle_mesh(output_mesh_dir, mesh)



    
    