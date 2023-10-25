#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 main_3d_eval.py --data_dir [data-path-to-h36m] --input_n 10 --output_n 25 --dct_n 35 --exp running_res --is_load


#cmu
CUDA_VISIBLE_DEVICES=0 python3 main_cmu_3d_eval.py --data_dir_cmu [data-path-to-cmu] --input_n 10 --output_n 25 --dct_n 35 --exp running_res_cmu --is_load

#3dpw
CUDA_VISIBLE_DEVICES=0 python3 main_3dpw_3d_eval.py --data_dir_3dpw [data-path-to-3dpw] --input_n 10 --output_n 30 --dct_n 40 --exp running_res_3dpw --is_load