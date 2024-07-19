#train
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48
#test
CUBLAS_WORKSPACE_CONFIG=:4096:8 python test.py --model_pth ./log/snapshot/model.pth