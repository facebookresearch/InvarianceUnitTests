# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
import glob
import os
import json
import argparse
import matplotlib.pyplot as plt
import plot_results

def print_row(row, col_width=15, latex=False):
    sep = " & " if latex else "  "
    end_ = "\\\\" if latex else ""
    print(sep.join([x.ljust(col_width) for x in row]), end_)


def print_table(table, col_width=15, latex=False):
    col_names = sorted(table[next(iter(table))].keys())
    
    print("\n")
    if latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\begin{table}")
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * len(col_names) + "}")
        print("\\toprule")

    print_row([""] + col_names, col_width, latex)

    if latex:
        print("\\midrule")

    for row_k, row_v in sorted(table.items()):
        row_values = [row_k]
        for col_k, col_v in sorted(row_v.items()):
            row_values.append(col_v)
        print_row(row_values, col_width, latex)

    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")
        print("\\label{main_results}")
        print("\\caption{Main results.}")
        print("\\end{table}")
        print("\\end{document}")


def print_table_hparams(table, col_width=15, latex=False):
    print("\n")
    for dataset in table.keys(): 
        print(dataset, "\n")
        for model in table[dataset].keys():
            print(model, table[dataset][model])
        print("\n")


def build_table(dirname, models=None, n_envs=None, num_dim=None, latex=False, standard_error=False):
    records = []
    for fname in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(fname, "r") as f:
            if os.path.getsize(fname) != 0:
                records.append(f.readline().strip())

    df = pd.read_json("\n".join(records), lines=True)
    if models is not None:
        df = df.query(f"model in {models}")
    if n_envs is not None:
        df = df.query(f"n_envs=={n_envs}")
    if num_dim is not None:
        df = df.query(f"num_dim=={num_dim}")

    print(f'{len(df)} records.')
    pm = "$\\pm$" if latex else "+-"

    table = {}
    table_avg = {}
    table_val = {}
    table_val_avg = {
        "data" : {},
        "n_envs": 0,
        "dim_inv": 0,
        "dim_spu": 0
    }
    table_hparams = {}

    for dataset in df["dataset"].unique():
        # filtered by dataset
        df_d = df[df["dataset"] == dataset]
        envs = sorted(list(set(
            [c[-1] for c in df_d.filter(regex="error_").columns])))
        if n_envs:
            envs = envs[:n_envs]

        table_hparams[dataset] = {}
        table_val[dataset] = {}
        for key in ["n_envs", "dim_inv", "dim_spu"]:
            table_val_avg[key] = int(df[key].iloc[0])
        table_val_avg["data"][dataset] = {}

        for model in df["model"].unique():
            # filtered by model
            df_d_m = df_d[df_d["model"] == model]

            best_model_seed = df_d_m.groupby("model_seed").mean().filter(
                regex='error_validation').sum(1).idxmin()

            # filtered by hparams
            df_d_m_s = df_d_m[df_d_m["model_seed"] == best_model_seed].filter(
                regex="error_test")

            # store the best hparams
            df_d_m_s_h = df_d_m[df_d_m["model_seed"] == best_model_seed].filter(
                regex="hparams")
            table_hparams[dataset][model] = json.dumps(
                df_d_m_s_h['hparams'].iloc[0])

            table_val[dataset][model] = {}
            for env in range(len(envs)):
                errors = df_d_m_s[["error_test_E" + str(env)]]
                std = float(errors.std(ddof=0))
                se = std / len(errors)
                fmt_str = "{:.2f} {} {:.2f}".format(
                    float(errors.mean()), pm, std)
                if standard_error:
                    fmt_str += " {} {:.1f}".format(
                        float('/', se))

                dataset_env = dataset + ".E" + str(env)
                if dataset_env not in table:
                    table[dataset_env] = {}

                table[dataset_env][model] = fmt_str
                table_val[dataset][model][env] = {
                    "mean": float(errors.mean()), 
                    "std": float(errors.std(ddof=0))
                    }

            # Avg
            if dataset not in table_avg:
                table_avg[dataset] = {}
            table_test_errors = df_d_m_s[["error_test_E" +
                                          str(env) for env in range(len(envs))]]
            mean = table_test_errors.mean(axis=0).mean(axis=0)
            std = table_test_errors.std(axis=0,ddof=0).mean(axis=0)
            table_avg[dataset][model] = f"{float(mean):.2f} {pm} {float(std):.2f}"
            table_val_avg["data"][dataset][model] = {
                "mean": float(mean), 
                "std":float(std),
                "hparams": table_hparams[dataset][model]
                }

    return table, table_avg, table_hparams, table_val, table_val_avg, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--num_dim', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    args = parser.parse_args()

    table, table_avg, table_hparams, table_val, table_val_avg, df = build_table(
        args.dirname, args.models, args.n_envs, args.num_dim, args.latex)

    # Print table and averaged table
    print_table(table, latex=args.latex)
    print_table(table_avg, latex=args.latex)

    # Print best hparams
    print_table_hparams(table_hparams)

    # Plot results
    commit = args.dirname.split('/')[-2]
    plot_results.plot_table(
        table=table_val, 
        dirname=args.dirname, 
        file_name='results_' + commit)
    plot_results.plot_table_avg(
        table=table_val_avg, 
        dirname=args.dirname, 
        file_name='results_avg_' + commit)
