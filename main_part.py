import argparse
import os
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="weights", help="Directory name to save the checkpoints [weights]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the output samples [samples]")

parser.add_argument("--train_bool", action="store_true", dest="train_bool", default=False, help="Training only bool with one network [False]")
parser.add_argument("--train_float", action="store_true", dest="train_float", default=False, help="Training only float with one network [False]")

parser.add_argument("--test_input", action="store", dest="test_input", default="", help="Select an input file for quick testing [*.sdf, *.binvox, *.ply, *.hdf5]")

parser.add_argument("--point_num", action="store", dest="point_num", default=4096, type=int, help="Size of input point cloud for testing [4096,16384,524288]")
parser.add_argument("--grid_size", action="store", dest="grid_size", default=64, type=int, help="Size of output grid for testing [32,64,128]")
parser.add_argument("--block_num_per_dim", action="store", dest="block_num_per_dim", default=5, type=int, help="Number of blocks per dimension [1,5,10]")
parser.add_argument("--block_padding", action="store", dest="block_padding", default=5, type=int, help="Padding for each block [5]")

parser.add_argument("--input_type", action="store", dest="input_type", default="sdf", help="Input type [sdf,voxel,udf,pointcloud,noisypc]")
parser.add_argument("--method", action="store", dest="method", default="ndc", help="Method type [ndc,undc,ndcx]")
parser.add_argument("--postprocessing", action="store_true", dest="postprocessing", default=False, help="Enable the post-processing step to close small holes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

import torch
import datasetpc
import modelpc
import utils

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

#Create network
CNN_3d = modelpc.local_pointnet_larger

#Create network
# receptive_padding = 3 #for grid input
pooling_radius = 2 #for pointcloud input
KNN_num = modelpc.KNN_num

network_bool = CNN_3d(out_bool=True, out_float=False)
network_bool.to(device)
network_float = CNN_3d(out_bool=False, out_float=True)
network_float.to(device)

def worker_init_fn(worker_id):
    np.random.seed(int(time.time()*10000000)%10000000 + worker_id)

#load weights
print('loading net...')
network_bool.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_bool.pth"))
print('network_bool weights loaded')
network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
print('network_float weights loaded')
print('loading net... complete')

#test
network_bool.eval()
network_float.eval()

#Create test dataset
dataset_test = datasetpc.scene_crop_pointcloud(FLAGS.test_input, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.block_num_per_dim, FLAGS.block_padding)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)  #batch_size must be 1

#create large grid，因为是大场景，往往有上百个 block
full_scene_size = np.copy(dataset_test.full_scene_size)
pred_output_bool_numpy = np.zeros([FLAGS.grid_size*full_scene_size[0],
                                    FLAGS.grid_size*full_scene_size[1],
                                    FLAGS.grid_size*full_scene_size[2], 3], np.int32)
pred_output_float_numpy = np.zeros([FLAGS.grid_size*full_scene_size[0],
                                    FLAGS.grid_size*full_scene_size[1],
                                    FLAGS.grid_size*full_scene_size[2], 3], np.float32)

full_size = full_scene_size[0]*full_scene_size[1]*full_scene_size[2]
for i, data in enumerate(dataloader_test, 0):
    print(i,"/",full_size)
    
    continue
    
    pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_ = data

    if pc_KNN_idx_.size()[1]==1: continue # 说明这个 block 没有内容，直接跳过去

    pc_KNN_idx      = pc_KNN_idx_[0].to(device)
    pc_KNN_xyz      = pc_KNN_xyz_[0].to(device)
    voxel_xyz_int   = voxel_xyz_int_[0].to(device)
    voxel_KNN_idx   = voxel_KNN_idx_[0].to(device)
    voxel_KNN_xyz   = voxel_KNN_xyz_[0].to(device)
    
    #  由 i 得到 idx_x, idx_y, idx_z, 详见 https://www.notion.so/qixuema/f09e3cf7234c41de9da8566133896ca3?v=03843a9d747649a7839c92c91dfabf7f&p=5fef3d80938e4b32b017f53c34a84da0&pm=s
    # 从下往上（沿z轴），从左往右（沿y轴），从后往前（沿x轴），依次对所有的 block 进行重建
    idx_x   = i // (full_scene_size[1]*full_scene_size[2])
    idx_yz  = i % (full_scene_size[1]*full_scene_size[2])
    idx_y   = idx_yz // full_scene_size[2]
    idx_z   = idx_yz % full_scene_size[2]

    with torch.no_grad():
        pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
        pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

        # 疑问：这里为什么是 FLAGS.grid_size*2+1？
        pred_output_bool_grid = torch.zeros([FLAGS.grid_size*2+1,
                                                FLAGS.grid_size*2+1,
                                                FLAGS.grid_size*2+1,3],
                                            dtype=torch.int32, device=device)
        pred_output_float_grid = torch.full([FLAGS.grid_size*2+1,
                                                FLAGS.grid_size*2+1,
                                                FLAGS.grid_size*2+1,3], 0.5, device=device)

        pred_output_bool_grid[voxel_xyz_int[:,0],
                                voxel_xyz_int[:,1],
                                voxel_xyz_int[:,2]] = (pred_output_bool > 0.3).int() # 这里可以修改阈值大小，默认是 0.3
        
        pred_output_float_grid[voxel_xyz_int[:,0],
                                voxel_xyz_int[:,1],
                                voxel_xyz_int[:,2]] = pred_output_float

        if FLAGS.postprocessing:
            pred_output_bool_grid = modelpc.postprocessing(pred_output_bool_grid) # [65, 65, 65, 3]

        pred_output_bool_numpy[idx_x * FLAGS.grid_size : (idx_x+1) * FLAGS.grid_size, 
                                idx_y * FLAGS.grid_size : (idx_y+1) * FLAGS.grid_size, 
                                idx_z * FLAGS.grid_size : (idx_z+1) * FLAGS.grid_size] \
            = pred_output_bool_grid[FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size,
                                    FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size,
                                    FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size].detach().cpu().numpy()
        
        pred_output_float_numpy[idx_x * FLAGS.grid_size : (idx_x+1) * FLAGS.grid_size, 
                                idx_y * FLAGS.grid_size : (idx_y+1) * FLAGS.grid_size, 
                                idx_z * FLAGS.grid_size : (idx_z+1) * FLAGS.grid_size] \
            = pred_output_float_grid[FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size,
                                        FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size,
                                        FLAGS.block_padding : FLAGS.block_padding + FLAGS.grid_size].detach().cpu().numpy()
                        
pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)


import cutils
vertices, triangles = cutils.dual_contouring_undc(
    np.ascontiguousarray(pred_output_bool_numpy, np.int32), 
    np.ascontiguousarray(pred_output_float_numpy, np.float32))

# mesh = utils.mesh_reorient(dataset_test, vertices, triangles)
utils.mesh_reorient(dataset_test, vertices, triangles)