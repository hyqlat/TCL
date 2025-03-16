#[-------h36m-------]
##train
python main_h36m_3d.py --data_dir [data-path-to-pgbig] --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 --skip_rate 1 --batch_size 16 --test_batch_size 32 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.001 --epoch 120 --test_sample_num -1

##test
python main_h36m_3d.py --data_dir [data-path-to-pgbig] --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 --skip_rate 1 --test_batch_size 32 --in_features 66 --cuda_idx cuda:0 --d_model 16 --test_sample_num -1 --lr_now 0.001 --is_eval

#[-------3dpw-------]
##train
python main_3dpw_3d.py --data_dir [data-path-to-3dpw] --kernel_size 10 --dct_n 40 --input_n 10 --output_n 30 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 69 --cuda_idx cuda:0 --d_model 16 --lr_now 0.001 --epoch 50 --test_sample_num -1
##test
python main_3dpw_3d.py --data_dir [data-path-to-3dpw]  --kernel_size 10 --dct_n 40 --input_n 10 --output_n 30 --skip_rate 1 --test_batch_size 32 --in_features 69 --cuda_idx cuda:0 --d_model 16 --test_sample_num -1 --lr_now 0.001 --is_eval

#[-------cmu-------]
###rain
python main_cmu_3d.py --data_dir [data-path-to-cmu] --kernel_size 10 --dct_n 35 --dct_n_lay5 15 --input_n 10 --output_n 25 --skip_rate 1 --batch_size 16 --test_batch_size 32 --in_features 75 --cuda_idx cuda:0 --d_model 16 --lr_now 0.001 --epoch 50 --test_sample_num -1
##test
python main_cmu_3d.py --data_dir [data-path-to-cmu]  --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 --skip_rate 1 --test_batch_size 32 --in_features 75 --cuda_idx cuda:0 --d_model 16 --test_sample_num -1 --lr_now 0.001 --is_eval


