import numpy as np
from statistics import mean
import open3d as o3d

def mesh_reorient(dataset_pc, vertices, triangles):
    # 恢复 vertices 的原始尺寸
    vertices = vertices  / (dataset_pc.output_grid_size * dataset_pc.block_num_per_dim) * dataset_pc.pc_scale + dataset_pc.pc_min

    # 然后，我们构建一个 KNN 树. 这里注意，必须要用 o3d 提供的点云数据类型作为输入来进行 kd 树构建，不能用 numpy 数据类型来构建，否则下面在 knn 搜寻的时候会报错
    point_cloud_data = dataset_pc.input_point
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_data) 
    
    vertices_size = len(vertices)
    vertices_normal = np.zeros_like(vertices)
    input_normals = np.asarray(point_cloud_data.normals)
    for i in range(0, vertices_size):
        # find 10 points that closed to point_temp
        [k, idx, _] = pcd_tree.search_knn_vector_3d(vertices[i], 50)
        # 然后利用KNN 计算每一个 vertex 周围 K 个点云的法向量
        neighbor_normals = input_normals[idx]
        # 对这K 个法向量进行取平均
        mean_normal=neighbor_normals.mean(axis=0)
        # 将平均值赋值给当前的vertex
        vertices_normal[i] = mean_normal
        
    # 修改 triangles 的顶点顺序
    for ii in range(len(triangles)):
        # reverse the order of triangle
        A_idx = triangles[ii,0]
        B_idx = triangles[ii,1]
        C_idx = triangles[ii,2]

        A_n = vertices_normal[A_idx]
        B_n = vertices_normal[B_idx]
        C_n = vertices_normal[C_idx]
        # 计算 A、B、C三个点的法向量均值
        mean_n = (A_n + B_n + C_n)/3
        
        A = vertices[A_idx]
        B = vertices[B_idx]
        C = vertices[C_idx]
        ab = B - A
        bc = C - B

        face_n = np.cross(ab, bc)
        res = (face_n* mean_n).sum()
        
        if res < 0:
            tmp = triangles[ii,1]
            triangles[ii,1] = triangles[ii,2]
            triangles[ii,2] = tmp
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    output_mesh_path = "./samples/reorient_mesh.ply"
    o3d.io.write_triangle_mesh(output_mesh_path, mesh, write_ascii=True)
    return mesh