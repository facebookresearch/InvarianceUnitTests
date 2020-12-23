# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/bash

if [ $1 == 'test' ]
then
    num_data_seeds=1
    num_model_seeds=1
    num_iterations=1
    dir_results='test_results'
else
    num_data_seeds=50
    num_model_seeds=20
    num_iterations=10000
    dir_results='test_results'
fi

### Default experiment ###
dim_inv=5
dim_spu=5
n_envs=3
echo "Default experiment: dim_inv=${dim_inv} dim_spu=${dim_spu} n_envs=${n_envs}"

python3.8 scripts/sweep.py \
    --models ERM IRMv1 ANDMask IGA Oracle \
    --num_iterations $num_iterations \
    --datasets Example1 Example1s Example2 Example2s Example3 Example3s \
    --dim_inv $dim_inv --dim_spu $dim_spu \
    --n_envs $n_envs \
    --num_data_seeds $num_data_seeds --num_model_seeds $num_model_seeds \
    --output_dir ${dir_results}/default/sweep_linear_nenvs=${n_envs}_dinv=${dim_inv}_dspu=${dim_spu} \
    --cluster \
    --jobs_cluster 200

### Varying the number of environments ###
dim_inv=5
dim_spu=5
echo "Varying number of environments: n_envs"

for n_envs in 2 3 4 5 6 7 8 9 10 
do
    echo "dim_inv=${dim_inv} dim_spu=${dim_spu} n_envs=${n_envs}"
    python3.8 scripts/sweep.py \
        --models ERM IRMv1 ANDMask IGA Oracle \
        --num_iterations 10 \
        --datasets Example1 Example1s Example2 Example2s Example3 Example3s \
        --dim_inv $dim_inv --dim_spu $dim_spu \
        --n_envs $n_envs \
        --num_data_seeds $num_data_seeds --num_model_seeds $num_model_seeds \
        --output_dir ${dir_results}/nenvs/sweep_linear_nenvs=${n_envs}_dinv=${dim_inv}_dspu=${dim_spu} \
        --cluster \
        --jobs_cluster 200
done

### Varying the spurious dimensions ###
dim_inv=5
n_envs=3
echo "Varying spurious dimensions: dim_spu"
for dim_spu in 0 1 2 3 4 5 6 7 8 9 10
do
    echo "dim_inv=${dim_inv} dim_spu=${dim_spu} n_envs=${n_envs}"
    python3.8 scripts/sweep.py \
        --models ERM IRMv1 ANDMask IGA Oracle \
        --num_iterations $num_iterations \
        --datasets Example1 Example1s Example2 Example2s Example3 Example3s \
        --dim_inv $dim_inv --dim_spu $dim_spu \
        --n_envs $n_envs \
        --num_data_seeds $num_data_seeds --num_model_seeds $num_model_seeds \
        --output_dir ${dir_results}/dimspu/sweep_linear_nenvs=${n_envs}_dinv=${dim_inv}_dspu=${dim_spu} \
        --cluster \
        --jobs_cluster 200
done 
