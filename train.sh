export WANDB_API_KEY="e22d14a8d4b7c0b3fd491180f4b4eb0d6c5114a2"
export NUM_WORKERS=16
export CUDA_VISIBLE_DEVICES=0,1

python3 src/train.py \
    experiment=ttda \
    model/net=gta5_source \
    datamodule/test_list=bdd