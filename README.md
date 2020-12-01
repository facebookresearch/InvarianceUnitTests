# Linear unit-tests for invariance discovery - Code

### Installing requirements

```bash
$ conda create -n causal python=3.8
$ conda activate causal
$ pip3 install -U -r requirements.txt
```

### Running the experiments and printing results

```bash
$ python sweep.py --num_iterations 1 --output_dir results/
$ python collect_results.py results/COMMIT
```

### Reproducing the paper figures

```bash
$ bash reproduce_plots.sh
```

### Reproducing the paper results (requires a cluster)
Careful, this launches 630 000 jobs for hyper-parameter search.

```bash
$ bash reproduce_results.sh test
```

### Deactivating and removing the env

```bash
$ conda deactivate
$ conda remove --name causal --all
```

## License

This source code is released under the MIT license, included [here](LICENSE). 
