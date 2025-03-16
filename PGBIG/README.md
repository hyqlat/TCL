# PGBIG+TCL

Please redirect to [PGBIG](https://github.com/705062791/PGBIG) for more details on preparing the dataset and environment.

#### Usage

Run the `train/test` in script `run.sh` for each dataset. For example:

```bash
#[-------h36m-------]
##train
python main_h36m_3d.py --data_dir [data-path-to-pgbig] --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 --skip_rate 1 --batch_size 16 --test_batch_size 32 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.001 --epoch 120 --test_sample_num -1
```

The code is partially based on [PGBIG](https://github.com/705062791/PGBIG).
