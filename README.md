# Linear unit-tests for invariance discovery - Code

Official code for the paper [Linear unit-tests for invariance discovery](https://arxiv.org/abs/2102.10867), presented as a spotlight talk at the [NeurIPS 2020 Workshop Causal Discovery & Causality-Inspired Machine Learning](https://www.cmu.edu/dietrich/causality/neurips20ws/).

### Installing requirements

```bash
conda create -n invariance python=3.8
conda activate invariance
python3.8 -m pip install -U -r requirements.txt
```

### Running a single experiment

```bash
python3.8 scripts/main.py \
    --model ERM --dataset Example1 --n_envs 3 \
    --num_iterations 10000 --dim_inv 5 --dim_spu 5 \
    --hparams '{"lr":1e-3, "wd":1e-4}' --output_dir results/
```

### Running the experiments and printing results

```bash
python3.8 scripts/sweep.py --num_iterations 10000 --num_data_seeds 1 --num_model_seed 1 --output_dir results/
python3.8 scripts/collect_results.py results/COMMIT
```

### Reproducing the figures

```bash
bash reproduce_plots.sh
```

### Reproducing the results (requires a cluster)

Be careful, this script launches 630 000 jobs for the hyper-parameter search.

```bash
bash reproduce_results.sh test
```

### Deactivating and removing the env

```bash
conda deactivate
conda remove --name invariance --all
```

## License

This source code is released under the MIT license, included [here](LICENSE).

## Reference

If you make use of our suite of tasks in your research, please cite the following in your manuscript:

```
@article{aubin2021linear,
  title={Linear unit-tests for invariance discovery},
  author={Aubin, Benjamin and S{\l}owik, Agnieszka and Arjovsky, Martin and Bottou, Leon and Lopez-Paz, David},
  journal={arXiv preprint arXiv:2102.10867},
  year={2021}
}
```
