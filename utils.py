import numpy as np
from statistics import mean
import open3d as o3d
from multiprocessing import Process, Queue
import queue

def get_normal(q, input_normals, pcd_tree, vertices_normal_part, list_of_vertices):
    # global vertices_normal
    vertices_num = len(list_of_vertices)
    for vid in range(vertices_num):
        pid = list_of_vertices[vid][0]
        idx = list_of_vertices[vid][1]
        vertice = list_of_vertices[vid][2]

        # find 10 points that closed to point_temp
        [k, idces, _] = pcd_tree.search_knn_vector_3d(vertice, 10)
        # 然后利用KNN 计算每一个 vertex 周围 K 个点云的法向量
        neighbor_normals = input_normals[idces]
        # 对这K 个法向量进行取平均
        mean_normal = neighbor_normals.mean(axis=0)
        # 将平均值赋值给当前的vertex
        # vertices_normal_dict[str(pid)][vid] = mean_normal
        vertices_normal_part[idx] = mean_normal
        
        # vertices_normal[idx] = mean_normal
    
        # q.put([1, pid, idx])
    q.put([1, pid, vertices_normal_part])


def mesh_reorient(dataset_pc, vertices, triangles): 
    
    # 恢复 vertices 的原始尺寸
    if dataset_pc.by_voxel_size:
        vertices = vertices * dataset_pc.voxel_size + dataset_pc.pc_min
    else:
        vertices = vertices  / (dataset_pc.output_grid_size * dataset_pc.block_num_per_dim) * dataset_pc.pc_scale + dataset_pc.pc_min
        
    reorient_flag = True # NOTE 因为这个面片重定向很耗时间，我也没优化，为了其他测试，我暂时先把它关掉了
    if reorient_flag:
        # 然后，我们构建一个 KNN 树. 这里注意，必须要用 o3d 提供的点云数据类型作为输入来进行 kd 树构建，不能用 numpy 数据类型来构建，否则下面在 knn 搜寻的时候会报错
        point_cloud_data = dataset_pc.input_point
        
        # TODO: 然后对点云进行降采样
        
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_data) 
        input_normals = np.asarray(point_cloud_data.normals)
        
        #prepare list of vertices
        even_distribution = [16]
        this_machine_id = 0
        num_of_process = 0
        P_start = 0
        P_end = 0
        for i in range(len(even_distribution)):
            num_of_process += even_distribution[i]
            if i < this_machine_id:
                P_start += even_distribution[i]
            if i <= this_machine_id:
                P_end += even_distribution[i]
        print(this_machine_id, P_start, P_end)
        
        list_of_list_of_vertices = []
        for i in range(num_of_process):
            list_of_list_of_vertices.append([])

        vertices_num = len(vertices)
        for idx in range(vertices_num):
            process_id = idx % num_of_process
            list_of_list_of_vertices[process_id].append([process_id, idx, vertices[idx]])
        
        #map processes
        q = Queue()
        workers = []
        vertices_normal_part = np.zeros([vertices_num, 3]).astype(np.float32)

        for i in range(P_start,P_end):
            list_of_vertices = list_of_list_of_vertices[i]
            workers.append(Process(target=get_normal, args = (q, input_normals, pcd_tree, vertices_normal_part, list_of_vertices)))

        for p in workers:
            p.start()

        # 这里我们用一个三维的 array 来承接多进程的输出结果
        vertices_normal_q = np.zeros([num_of_process, vertices_num, 3]).astype(np.float32)
        for i in range(num_of_process):
            _, pid, vertices_normal_part = q.get()
            vertices_normal_q[pid] = vertices_normal_part

        vertices_normal = vertices_normal_q.sum(axis=0)
        
        print("finished knn")
        
        
        # 修改 triangles 的顶点顺序
        # 获得面片三个顶点的平均法向
        triangles_num = len(triangles)
        triangles_np = np.array(triangles)
        triangles_np = triangles_np.flatten()
        triangles_normal = vertices_normal[triangles_np]
        triangles_normal = triangles_normal.reshape(triangles_num, 3, 3)
        mean_n = triangles_normal.mean(axis=1)
        
        # 获得面片的法向
        vertices_triangles = vertices[triangles_np]
        vertices_triangles = vertices_triangles.reshape(triangles_num, 3, 3)
        ab = vertices_triangles[:,1] - vertices_triangles[:,0]
        bc = vertices_triangles[:,2] - vertices_triangles[:,1]
        face_n = np.cross(ab, bc)
        
        # 根据点乘结果对面片的方向进行调整
        res = (face_n * mean_n).sum(axis=1)
        idx = np.argwhere(res < 0)
        tmp = triangles[idx,1]
        triangles[idx,1] = triangles[idx,2]
        triangles[idx,2] = tmp
        
        print("finished reoriention")
             
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    if reorient_flag:
        output_mesh_path = "./samples/" + dataset_pc.file_name +"_reorient_mesh.ply"
    else:
        output_mesh_path = "./samples/" + dataset_pc.file_name +"_undc_mesh.ply"
        
    o3d.io.write_triangle_mesh(output_mesh_path, mesh, write_ascii=True)
    return mesh