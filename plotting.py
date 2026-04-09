import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl

def plot_fcs(fcs, names, neworder, save_path):
    
    fig, axes = plt.subplots(2, 2, figsize = (25 ,25))
    axes = axes.flatten()
    
    for i, (ax, fc, name) in enumerate(zip(axes, fcs, names)):
        fc = fc.to_numpy()

        sns.heatmap(
            fc[neworder].T[neworder].T, 
            ax=ax, 
            vmin=-1.0, 
            vmax=1.0, 
            cmap='RdBu_r', 
            cbar=False, 
            square=True, 
            xticklabels=False, 
            yticklabels=False
        )
        
        ax.set_title(name, fontsize=40)
    
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=30, labelpad=0)

    plt.subplots_adjust(right=0.9)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path)



if __name__ == '__main__':
    
    mpl.rcParams["axes.labelsize"] = 20    # x and y axis labels
    mpl.rcParams["xtick.labelsize"] = 18   # x tick labels
    mpl.rcParams["ytick.labelsize"] = 18   # y tick labels
    mpl.rcParams["axes.titlesize"] = 24    # plot title
    mpl.rcParams["font.size"] = 20         # base font size
    
    
    
    df = pd.read_csv('results/final_results/transformer_noise_again.csv')
    df.insert(0, 'Model', ['Transformer Single Subject'] * df.shape[0])
    df1 = pd.read_csv('results/final_results/baseline_fixed.csv')
    df1.insert(0, 'Model', ['Baseline'] * df1.shape[0])
    df2 = pd.read_csv('results/final_results/transformer_all_ss2.csv')
    df2.insert(0, 'Model', 'Transformer Fine Tuned')
    
    subs = df2['Scan'].tolist()
    
    df = pd.concat([df, df1, df2], ignore_index=True)
    df = df[df['Scan'].isin(subs)]
    df.to_csv('all_models.csv')
    
    plt.figure(figsize=(12, 10))
    sns.boxplot(
        data=df,
        x='Model',
        y='r',
        order=['Baseline', 'Transformer Single Subject', 'Transformer Fine Tuned']
    )
    plt.ylabel('Pearson Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('pcc.png')
    
    df = df[df['recon_err'] < 10]
    
    plt.figure(figsize=(12, 10))
    sns.boxplot(
        data=df,
        x='Model',
        y='recon_err',
        order=['Baseline', 'Transformer Single Subject', 'Transformer Fine Tuned']
    )
    plt.ylabel('Mean Squared Reconstruction Error')
    plt.tight_layout()
    plt.savefig('recon.png')
    
        
    transformer_dir = Path('results/final_results/transformer_fixed_noise_again')
    baseline_dir = Path('results/final_results/baseline')
    trans_all_dir = Path('results/final_results/transformer_all_ss2')
    fig_dir = Path('results/final_results/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    
    with open('region_labels.txt', 'r') as f:
        labels = f.readlines()
    labels = [x.removesuffix('\n') for x in labels]
    neworderlr = np.loadtxt('./neworderlr.txt')
    neworder = np.array([np.where(neworderlr == i)[0][0] for i in range(len(neworderlr))])
    
    df = pd.read_csv('results/final_results/all_models.csv', index_col=0)
        
    for sub in trans_all_dir.iterdir():
        sub_df = df[df['Scan'] == sub.name]
        print(sub_df)
        sub_name = sub.name
        real_fc = pd.read_csv(
            sub.joinpath('connectomes', 'real.csv'),
            index_col=0
        )
        trans_ss_fc = pd.read_csv(
            transformer_dir.joinpath(sub_name, 'connectomes', 'model.csv'),
            index_col=0
        )
        baseline_ss_fc = pd.read_csv(
            baseline_dir.joinpath(sub_name, 'connectomes', 'model.csv'),
            index_col=0
        )
        trans_all_fc = pd.read_csv(
            sub.joinpath('connectomes', 'model.csv'),
            index_col=0
        )
        plot_fcs(
            [real_fc, trans_ss_fc, baseline_ss_fc, trans_all_fc], 
            ['Real FC', 'Transformer Single Subject FC', 'Baseline FC', 'Transformer Fine-Tuned FC'], 
            neworder,
            fig_dir.joinpath(sub_name + '.png')
        )
        
    
        

        