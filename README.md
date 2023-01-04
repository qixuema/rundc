
## Requirements
- Python 3 with numpy, Cython, open3d（I use Pyhton=3.7）
- [PyTorch 1.8](https://pytorch.org/get-started/locally/) (other versions also work)

Build Cython module:
```
python setup.py build_ext --inplace
```

To test on a large scene (noisy point cloud input):
```
python main.py --test_input examples/input_pointcloud.ply --input_type noisypc --method undc --postprocessing --point_num 524288 --grid_size 64 --block_num_per_dim 10
```
Note that the code will crop the entire scene into overlapping patches.
 ```--point_num``` specifies the maximum number of input points per patch. 
 ```--grid_size``` specifies the size of the output grid per patch. 
 ```--block_padding``` controls the boundary padding for each patch to make the patches overlap with each other so as to avoid seams; the default value is good enough in most cases. 
 ```--block_num_per_dim``` specifies how many crops the scene will be split into. In the above command, the input point cloud will be normalized into a cube and the cube will be split into 10x10x10 patches (although some patches are empty).