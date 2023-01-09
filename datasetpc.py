import os
import numpy as np
import torch
import open3d as o3d
import math
from sklearn.neighbors import KDTree

#only for testing
class scene_crop_pointcloud(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_point_num, output_grid_size, KNN_num, pooling_radius, block_num_per_dim, block_padding):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.block_num_per_dim = block_num_per_dim
        self.block_padding = block_padding
        self.by_voxel_size = True
        self.voxel_size = 0.1 # NOTE 这里我们添加一个新的变量，从而我们可以根据 voxel_size 大小来确定 block 的数量

        if self.data_dir.split(".")[-1]=="ply":
            point_cloud_data = o3d.io.read_point_cloud(self.data_dir)
            LOD_input = np.asarray(point_cloud_data.points).astype(np.float32)
        else:
            print("ERROR: invalid input type - only support ply or hdf5")
            exit(-1)

        
        #normalize to unit cube for each crop
        LOD_input_min = np.min(LOD_input,0)
        LOD_input_max = np.max(LOD_input,0)
        LOD_input_scale = np.max(LOD_input_max-LOD_input_min) 
        LOD_input = LOD_input-np.reshape(LOD_input_min, [1,3])
        
        if self.by_voxel_size:
            LOD_input = LOD_input/(self.output_grid_size * self.voxel_size)
        else:
            LOD_input = LOD_input/(LOD_input_scale/self.block_num_per_dim)
            
        self.full_scene = LOD_input
        self.full_scene_size = np.ceil(np.max(self.full_scene,0)).astype(np.int32)
        print("Crops:", self.full_scene_size)
        self.full_scene = self.full_scene*self.output_grid_size
        
        # 用于 reorientation
        self.input_point = point_cloud_data
        self.pc_scale = LOD_input_scale
        self.pc_min = LOD_input_min
        
        # 用于重命名输出文件
        _, self.file_name = os.path.split(self.data_dir)
        self.file_name = self.file_name[0:-4]


    def __len__(self):
        return self.full_scene_size[0]*self.full_scene_size[1]*self.full_scene_size[2]

    def __getitem__(self, index):
        grid_size = self.output_grid_size+self.block_padding*2

        idx_x = index//(self.full_scene_size[1]*self.full_scene_size[2])
        idx_yz = index%(self.full_scene_size[1]*self.full_scene_size[2])
        idx_y = idx_yz//self.full_scene_size[2]
        idx_z = idx_yz%self.full_scene_size[2]

        gt_input_mask_ = (self.full_scene[:,0]>idx_x*self.output_grid_size-self.block_padding) & (self.full_scene[:,0]<(idx_x+1)*self.output_grid_size+self.block_padding) & (self.full_scene[:,1]>idx_y*self.output_grid_size-self.block_padding) & (self.full_scene[:,1]<(idx_y+1)*self.output_grid_size+self.block_padding) & (self.full_scene[:,2]>idx_z*self.output_grid_size-self.block_padding) & (self.full_scene[:,2]<(idx_z+1)*self.output_grid_size+self.block_padding)

        if np.sum(gt_input_mask_)<100:
            return np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32)
        
        gt_input_ = self.full_scene[gt_input_mask_] - np.array([[idx_x*self.output_grid_size-self.block_padding,idx_y*self.output_grid_size-self.block_padding,idx_z*self.output_grid_size-self.block_padding]], np.float32)

        np.random.shuffle(gt_input_)
        gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)

        #write_ply_point(str(index)+".ply", gt_input_)

        pc_xyz = gt_input_
        kd_tree = KDTree(pc_xyz, leaf_size=8)
        pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)
        pc_KNN_idx = np.reshape(pc_KNN_idx,[-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz),self.KNN_num,3]) - np.reshape(pc_xyz,[len(pc_xyz),1,3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz)*self.KNN_num,3])
        #this will be used to group point features
        
        #consider all grid cells within range to the point cloud
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int,0,grid_size)
        tmp_grid = np.zeros([grid_size+1,grid_size+1,grid_size+1], np.uint8)
        tmp_grid[pc_xyz_int[:,0],pc_xyz_int[:,1],pc_xyz_int[:,2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1,1:-1,1:-1])
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k] = tmp_mask | tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k]
        voxel_x,voxel_y,voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate([np.reshape(voxel_x,[-1,1]),np.reshape(voxel_y,[-1,1]),np.reshape(voxel_z,[-1,1])],1)
        voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)
            
        voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx,[-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz),self.KNN_num,3]) - np.reshape(voxel_xyz,[len(voxel_xyz),1,3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz)*self.KNN_num,3])

        return pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz
    
