# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pandas as pd
import glob
import os
import json
import argparse
import matplotlib.pyplot as plt
import collect_results
import numpy as np
import torch

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size=14)


def plot_table(table, dirname, file_name, save=True, block=False, fontsize=12):
    fig, axs = plt.subplots(1, 6, figsize=(13, 2.1))
    axs = axs.flat
    width = None
    for id_d, dataset in zip(range(len(table.keys())), sorted(table.keys())):
        models = table[dataset]
        envs = models[list(models.keys())[0]].keys()
        if not width:
            width = 1 / (len(envs) + 1)
        legends = []
        for id_e, env in zip(range(len(envs)), envs):
            labels = sorted(models.keys())
            pos = np.arange(len(labels))
            model_means = [models[model][env]['mean']
                           for model in sorted(models.keys())]
            model_stds = [models[model][env]['std']
                           for model in sorted(models.keys())]
            l = axs[id_d].bar(pos + id_e * width, model_means, 
                        width=width, color=f'C{id_e}', label=f'E{env}',
                        align='center', ecolor=f'black', capsize=3, yerr=model_stds,
                        )
            legends.append(l)

        axs[id_d].set_title(dataset)
        axs[id_d].set_xticks(pos + width * (len(envs) / 2 - 0.5))
        axs[id_d].set_xticklabels(labels, fontsize=7)
        axs[id_d].set_ylim(bottom=0)


    axs[0].set_ylabel('Test error')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.legend(handles=legends,
            ncol=6,
            loc="lower center",
            bbox_to_anchor=(-2.8, -0.4))

    if save:
        fig_dirname = "figs/"
        os.makedirs(fig_dirname, exist_ok=True)
        models = '_'.join(sorted(models.keys()))
        plt.savefig(fig_dirname + file_name + '_' + models +'.pdf',
                    format='pdf', bbox_inches='tight')

    if block:
        plt.show(block=False)
        input('Press to close')
        plt.close('all')


def plot_table_avg(table, dirname, file_name, save=True, block=False, fontsize=12):
    table = table["data"]

    fig, axs = plt.subplots(1, 6, figsize=(13, 2.1))
    axs = axs.flat
    width = 0.5
    for id_d, dataset in zip(range(len(table.keys())), sorted(table.keys())):
        models = table[dataset]
        labels = sorted(models.keys())
        pos = np.arange(len(labels))
        model_means = [models[model]['mean']
                       for model in sorted(models.keys())]
        model_stds = [models[model]['std']
                       for model in sorted(models.keys())]
        legends = []
        for id_m in range(len(pos)):
            l, = axs[id_d].bar(pos[id_m], model_means[id_m], 
                        width=width, color=f'C{id_m}',
                        align='center', ecolor='black', 
                        capsize=7, yerr=model_stds[id_m], linewidth=0.1
                        )
            legends.append(labels[id_m])

        axs[id_d].set_title(dataset)
        axs[id_d].set_xticks(pos)
        axs[id_d].set_ylim(bottom=0)

    axs[0].set_ylabel('Test error')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.legend(legends,
            ncol=6,
            loc="lower center",
            bbox_to_anchor=(-2.8, -0.5))

    if save:
        fig_dirname = "figs/"
        os.makedirs(fig_dirname, exist_ok=True)
        models = '_'.join(sorted(models.keys()))
        plt.savefig(fig_dirname + file_name + '_' + models +'.pdf',
                    format='pdf', bbox_inches='tight')
    if block:
        plt.show(block=False)
        input('Press to close')
        plt.close('all')


def plot_nenvs_dimspu(df_nenvs, df_dimspu, dirname, file_name='', save=True, block=False):
    fig, axs = plt.subplots(2, 6, figsize=(13, 5))
    axs = axs.flat
    counter = 0 
    # Top: nenvs
    datasets = df_nenvs["dataset"].unique()
    for id_d, dataset in zip(range(len(datasets)), sorted(datasets)):
        df_d =  df_nenvs[df_nenvs["dataset"] == dataset]
        models = df_d["model"].unique()
        legends = []
        for id_m, model in zip(range(len(models)), sorted(models)):
            df_d_m = df_d[df_d["model"] == model].sort_values(by="n_envs")
            legend, =  axs[id_d].plot(df_d_m["n_envs"]/5, df_d_m["mean"],
                color=f'C{id_m}',
                label=model,
                linewidth=2)
            top = (df_d_m["mean"]+df_d_m["std"]/2).to_numpy()
            bottom = (df_d_m["mean"]-df_d_m["std"]/2).to_numpy()
            xs = np.arange(2, 11) / 5
            axs[id_d].fill_between(xs, bottom, top, facecolor=f'C{id_m}', alpha=0.2)
            legends.append(legend)
        
        axs[id_d].set_xlabel(r'$\delta_{\rm env}$')
        axs[id_d].set_title(dataset)
        axs[id_d].set_ylim(bottom=-0.005)
        axs[id_d].set_xlim(left=0.4, right=2)
        counter += 1

    # Bottom: dimspu
    datasets = df_dimspu["dataset"].unique()
    for id_d, dataset in zip(range(counter, counter+len(datasets)), sorted(datasets)):
        df_d =  df_dimspu[df_dimspu["dataset"] == dataset]
        models = df_d["model"].unique()
        legends = []
        for id_m, model in zip(range(len(models)), sorted(models)):
            df_d_m = df_d[df_d["model"] == model].sort_values(by="dim_spu")
            legend, =  axs[id_d].plot(df_d_m["dim_spu"]/5, df_d_m["mean"],
                color=f'C{id_m}',
                label=model,
                linewidth=2)
            top = (df_d_m["mean"]+df_d_m["std"]/2).to_numpy()
            bottom = (df_d_m["mean"]-df_d_m["std"]/2).to_numpy()
            xs = np.arange(0, 11) / 5
            axs[id_d].fill_between(xs, bottom, top, facecolor=f'C{id_m}', alpha=0.2)
            legends.append(legend)
        
        axs[id_d].set_xlabel(r'$\delta_{\rm spu}$')
        axs[id_d].set_title(dataset)
        axs[id_d].set_ylim(bottom=-0.005)
        axs[id_d].set_xlim(left=0, right=2)


    axs[0].set_ylabel("Test error")
    axs[6].set_ylabel("Test error")
    plt.tight_layout(pad=0)
    plt.legend(handles=legends,
            ncol=6,
            loc="lower center",
            bbox_to_anchor=(-2.8, -0.7))

    if save:
        fig_dirname = "figs/" + dirname
        os.makedirs(fig_dirname, exist_ok=True)
        models = '_'.join(models)
        plt.savefig(fig_dirname + file_name + '.pdf',
                    format='pdf', bbox_inches='tight')
    if block:
        plt.show(block=False)
        input('Press to close')
        plt.close('all')


def build_df(dirname):
    df = pd.DataFrame(columns=['n_envs', 'dim_inv', 'dim_spu', 'dataset', 'model', 'mean', 'std'])
    for filename in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(filename) as f:
            dic = json.load(f)
            n_envs = dic["n_envs"]
            dim_inv = dic["dim_inv"]
            dim_spu = dic["dim_spu"]
            for dataset in dic["data"].keys():
                single_dic = {}
                for model in dic["data"][dataset].keys():
                    mean =  dic["data"][dataset][model]["mean"]
                    std = dic["data"][dataset][model]["std"]
                    single_dic = dict(
                        n_envs=n_envs,
                        dim_inv=dim_inv,
                        dim_spu=dim_spu,
                        dataset=dataset,
                        model=model,
                        mean=mean,
                        std=std
                        )
                    # print(single_dic)
                    df = df.append(single_dic, ignore_index=True)

    return df


def process_results(dirname, commit, save_dirname): 
    subdirs = [os.path.join(dirname, subdir, commit + '/') for subdir in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, subdir))]
    for subdir in subdirs:
        print(subdir)
        table, table_avg, table_hparams, table_val, table_val_avg, df = collect_results.build_table(subdir)

        # plot table_val
        plot_table(
                table=table_val, 
                dirname=subdir, 
                file_name='_'.join(subdir.split('/')[-3:-1]),
                save=True, block=False)
        # save table_val
        save_dirname_single = save_dirname + "single/"
        os.makedirs(save_dirname_single, exist_ok=True)
        results_filename = os.path.join(save_dirname_single, 'single_' + '_'.join(subdir.split('/')[-4:-1]) + ".jsonl")
        results_file = open(results_filename, "w")
        results_file.write(json.dumps(table_val))
        results_file.close()

        # plot table_val_avg
        plot_table_avg(
                table=table_val_avg, 
                dirname=subdir, 
                file_name='avg_' + '_'.join(subdir.split('/')[-3:-1]),
                save=True, block=False)
        # save table_val_avg
        save_dirname_avg = save_dirname + "avg/"
        os.makedirs(save_dirname_avg, exist_ok=True)
        results_filename = os.path.join(save_dirname_avg, 'avg_' + '_'.join(subdir.split('/')[-4:-1]) + ".jsonl")
        results_file = open(results_filename, "w")
        results_file.write(json.dumps(table_val_avg))
        results_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dirname")
    parser.add_argument("-commit")
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    dirname_nenvs = "results_processed/nenvs/" + args.commit + "/"
    dirname_dimspu = "results_processed/dimspu/" + args.commit + "/"

    # construct averaged data
    if not args.load: 
        process_results(dirname=args.dirname + "nenvs/", commit=args.commit, save_dirname=dirname_nenvs)   
        process_results(dirname=args.dirname + "dimspu/", commit=args.commit, save_dirname=dirname_dimspu)        
        
    # plot results for different number of envs 
    df_nenvs = build_df(dirname_nenvs + "avg/")
    df_dimspu = build_df(dirname_dimspu + "avg/")

    plot_nenvs_dimspu(
            df_nenvs=df_nenvs, 
            df_dimspu=df_dimspu, 
            dirname= args.dirname.split('/')[-1], 
            file_name= 'results_nenvs_dimspu_' + args.commit,
            save=True, block=False)

    dirname = dirname_nenvs + "avg/"
    file_name = "avg_nenvs_final_sweep_linear_nenvs=3_dinv=5_dspu=5_e717c2ff36"
    results_filename = os.path.join(dirname, file_name + ".jsonl")
    table_avg = json.load(open(results_filename, "r"))
    plot_table_avg(
            table=table_avg, 
            dirname='', 
            file_name=file_name,
            save=True, block=False)

    dirname = dirname_nenvs + "single/"
    file_name = "single_nenvs_final_sweep_linear_nenvs=3_dinv=5_dspu=5_e717c2ff36"
    results_filename = os.path.join(dirname, file_name + ".jsonl")
    table = json.load(open(results_filename, "r"))
    plot_table(
            table=table, 
            dirname='', 
            file_name=file_name,
            save=True, block=False)


    
