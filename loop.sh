#!/bin/bash

conda activate env
cd /mutations

for exp in uber-ga chsw
do
  for seed in 386 843 2495 3407 12345 65464 98736 216873 345712 4876523 7416458 8496541 59876431
  do
    for env in Swimmer-v4 Humanoid-v4
    do
      mpirun -np 48 python gym_experiment.py configs/$exp.json --force_env=$env --force_seed=$seed --force_obnorm=1 --force_lr=0.1
    done

    for env Hopper-v4 HalfCheetah-v4 Walker2d-v4 Ant-v4
    do
      mpirun -np 48 python gym_experiment.py configs/$exp.json --force_env=$env --force_seed=$seed --force_obnorm=1 --force_lr=0.01
    done
  done
done