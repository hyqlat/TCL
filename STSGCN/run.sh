
CUDA_VISIBLE_DEVICES=5 python main_amass_3d.py --input_n 10 --output_n 25 --skip_rate 5 --joints_to_consider 18 --gamma 0.8 --model_path ./checkpoints/CKPT_3D_AMASS --data_dir [path-to-amass] --mode test  