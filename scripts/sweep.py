# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import main
import random
import models
import datasets
import argparse
import getpass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--datasets', nargs='+', default=[])
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_data_seeds', type=int, default=50)
    parser.add_argument('--num_model_seeds', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--callback', action='store_true')
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--jobs_cluster', type=int, default=512)
    args = vars(parser.parse_args())

    try:
        import submitit
    except:
        args["cluster"] = False
        pass

    all_jobs = []
    if len(args["models"]):
        model_lists = args["models"]
    else:
        model_lists = models.MODELS.keys()
    if len(args["datasets"]):
        dataset_lists = args["datasets"]
    else:
        dataset_lists = datasets.DATASETS.keys()

    for model in model_lists:
        for dataset in dataset_lists:
            for data_seed in range(args["num_data_seeds"]):
                for model_seed in range(args["num_model_seeds"]):
                    train_args = {
                        "model": model,
                        "num_iterations": args["num_iterations"],
                        "hparams": "random" if model_seed else "default",
                        "dataset": dataset,
                        "dim_inv": args["dim_inv"],
                        "dim_spu": args["dim_spu"],
                        "n_envs": args["n_envs"],
                        "num_samples": args["num_samples"],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "output_dir": args["output_dir"],
                        "callback": args["callback"]
                    }

                    all_jobs.append(train_args)

    random.shuffle(all_jobs)

    print("Launching {} jobs...".format(len(all_jobs)))

    if args["cluster"]:
        executor = submitit.SlurmExecutor(
            folder=f"/checkpoint/{getpass.getuser()}/submitit/")
        executor.update_parameters(
            time=3*24*60,
            gpus_per_node=0,
            array_parallelism=args["jobs_cluster"],
            cpus_per_task=1,
            comment="",
            partition="learnfair")

        executor.map_array(main.run_experiment, all_jobs)
    else:
        for job in all_jobs:
            print(main.run_experiment(job))
