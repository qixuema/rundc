python main.py \
    --checkpoint_dir weights \
    --test_input examples/input_pointcloud.ply \
    --input_type noisypc \
    --method undc \
    --postprocessing \
    --point_num 524288 \
    --grid_size 64 \
    --block_num_per_dim 10 \
    --gpu 0
