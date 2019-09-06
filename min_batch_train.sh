#!/usr/bin/env bash
python min_batch_train.py --gpu_ids 0 1 --lr 0.005 --batchsize 64 --nb_heads_1 4 --nb_supports_1 16 --nb_heads_2 4 --nb_supports_2 8 --nb_heads_3 2 --nb_supports_3 4 --order1_attention --hidden 32 --weight_decay 0.0 --dropout 0.4 --num_basis 16 --dataset reddit --print_every 32 --epochs 10000 --patience 10
