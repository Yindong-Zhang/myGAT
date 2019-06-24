#!/usr/bin/env bash

dropout=(0.2 0.4 0.6)
weight_decay=(5E-4 1E-3 1E-4)

for d in ${dropout[*]}
do
python train_ppi.py --hidden 128 --nb_heads_1 6 --nb_heads_2 4 --lr 5E-4 --nheads_last 4 --weight_decay 5E-4 --patience 500 --print_every 1 --dropout $d --batch_size 4 --order2_attention --gpu_ids 0 1

done