import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
from omegaconf import OmegaConf
from pathlib import Path
from models import get_model
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data


def remove_diag(arr):

    return arr[~np.eye(arr.shape[0], dtype=bool)]


def sort_fn(path: Path):
    loss = path.name.split('_')[-1].removesuffix('.pt')
    return float(loss)


def find_best_loss(path):
    model = path.joinpath('models/final_model.pt')
    return model
    # models = [x for x in path.rglob('**/*best_val_loss*.pt')]
    # return sorted(models, key=sort_fn)[0]


def find_raw_data(path, name):
    return list(path.rglob(f'**/{name}*timeseries.csv'))[0]


def get_recon(model, real_data, steps):
    device = next(model.parameters()).device
    model.eval()
    outputs = [torch.tensor(real_data[x]).to(device) for x in range(steps)]
    for _ in range(steps, real_data.shape[0]):
        try:
            sim_input = torch.vstack(outputs[-steps:]).to(device=device, dtype=torch.float)
            outputs.append(
                model(sim_input.unsqueeze(0).to(device))
            )
        except:
            sim_input = torch.cat(outputs[-steps:]).to(device=device, dtype=torch.float)
            outputs.append(
                model(sim_input.unsqueeze(0).to(device)).squeeze(0)
            )
    outputs = torch.vstack(outputs).detach().cpu().numpy()
    
    return outputs, np.mean((real_data[steps+1:] - outputs[steps+1:]) ** 2)


def get_model_fc(model, steps, sim_data_length, num_regions):
    
    rng = np.random.default_rng(seed=42)
    
    device = next(model.parameters()).device
    model.eval()
    outputs = [torch.zeros(num_regions, device=device, dtype=torch.float).to(device) for _ in range(steps)]
    
    for _ in range(sim_data_length):
        
        try:
            noise = 0.1 * rng.standard_normal(size=(steps, num_regions))
            sim_input = torch.vstack(outputs[-steps:]).to(device) + torch.tensor(noise, dtype=torch.float).to(device)
            outputs.append(
                model(sim_input.unsqueeze(0).to(device))
            )
        except:
            noise = 0.1 * np.random.randn(steps * num_regions)
            sim_input = torch.cat(outputs[-steps:]).to(device) + torch.tensor(noise, dtype=torch.float).to(device)
            outputs.append(
                model(sim_input.unsqueeze(0).to(device)).squeeze(0)
            )
        
        
    outputs = torch.vstack(outputs)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=5, neginf=-5)

    try:
        connectome = ConnectivityMeasure(kind='correlation').fit_transform(outputs.detach().cpu().numpy())
    except ValueError:
        connectome = np.zeros((outputs.shape[1], outputs.shape[1]))
        return connectome
    return connectome[0]

    
    
def get_model_fc_gcn(model, steps, sim_data_length, num_regions, fc, threshold):
    
    src, des = np.where(np.abs(fc) > threshold)
    edge_idx = np.stack([src, des])
    weights = np.abs(fc[src, des])
    device = next(model.parameters()).device
    outputs = [torch.zeros(num_regions, device=device, dtype=torch.float).to(device) for _ in range(steps)]
    
    for _ in range(sim_data_length):
        noise = 0.1 * np.random.randn(steps, num_regions)
        sim_input = torch.vstack(outputs[-steps:]).to(device) + torch.tensor(noise, dtype=torch.float).to(device)
        sim_input = Data(
            x=sim_input[-steps:].t(),
            edge_index=torch.tensor(edge_idx, dtype=torch.long).to(device),
            edge_attr=torch.tensor(weights, dtype=torch.float).to(device),
            # y=torch.tensor(bold_data[step], dtype=torch.float).unsqueeze(-1)
        )
        outputs.append(
            model(sim_input.x, sim_input.edge_index, sim_input.edge_attr).t()
        )
    outputs = torch.vstack(outputs)
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=5, neginf=-5)

    try:
        connectome = ConnectivityMeasure(kind='correlation').fit_transform(outputs.detach().cpu().numpy())
    except ValueError:
        connectome = np.zeros((outputs.shape[1], outputs.shape[1]))
        return connectome
    return connectome[0]
    

if __name__ == '__main__':
    
    
    warnings.simplefilter('ignore', FutureWarning)
    
    
    work_dir = Path('results/final_results/baseline')
    try:
        df = pd.read_csv(f'{work_dir.name}.csv', index_col=0)
    except:
        df = pd.DataFrame(columns=['r', 'p'])
    
    # Organizing labels into networks
    with open('region_labels.txt', 'r') as f:
        labels = f.readlines()
    labels = [x.removesuffix('\n') for x in labels]
    neworderlr = np.loadtxt('./neworderlr.txt')
    neworder = np.array([np.where(neworderlr == i)[0][0] for i in range(len(neworderlr))])
    
    
    for run in tqdm(work_dir.iterdir(), total=len(list(work_dir.iterdir()))):
        if run in df.index.to_list():
            continue
        cfg = OmegaConf.load(run.joinpath('config.yaml'))
        model = get_model(
            cfg.model.name, 
            **cfg.model.kwargs 
        )
        try:
            model.load_state_dict(
                torch.load(find_best_loss(run))['model_state'],
            )
        except: 
            continue
        
        cfg.run_name = cfg.run_name.split('preproc')[0]
        bold_data = pd.read_csv(find_raw_data(Path('data_like-npi/hcp'), cfg.run_name), index_col=0).to_numpy().astype(np.float64)
        real_fc = ConnectivityMeasure(kind='correlation').fit_transform(bold_data[30:, :360])[0]
        model.eval()
        model.to('cuda:1')
        model_fc = get_model_fc(model, cfg.data.train.step, 1200, 360)
                
        recon, recon_err = get_recon(model, bold_data[int(0.8 * bold_data[30:].shape[0]):, :360], cfg.data.train.step)
        run.joinpath('recon_signal').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recon).to_csv(run.joinpath('recon_signal', 'signal.csv'))
        df.at[run.name, 'recon_err'] = recon_err
        
        
        r, p = pearsonr(remove_diag(real_fc), remove_diag(model_fc))
        df.at[run.name, 'r'] = r
        df.at[run.name, 'p'] = p
        plt.figure(figsize = (4.8, 4.8))
        plt.scatter(remove_diag(model_fc), remove_diag(real_fc))
        plt.xlim(-0.7, 1.1)
        plt.xticks([-0.5, 0.0, 1.0])
        plt.xlabel('model FC')
        plt.ylim(-0.7, 1.1)
        plt.yticks([-0.5, 0.0, 1.0])
        plt.ylabel('empirical FC')
        plt.text(-0.6, 0.9, 'r = {:.2f}, p = {:.0e}'.format(r, p))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        run.joinpath('figures', 'correlation_plot.png').parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(run.joinpath('figures', 'correlation_plot.png'))
        plt.close()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
        sns.heatmap(real_fc[neworder].T[neworder].T, ax = ax1, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
        sns.heatmap(model_fc[neworder].T[neworder].T, ax = ax2, vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r', cbar = False, square = True, xticklabels = False, yticklabels = False)
        ax1.set_title('Empirical FC')
        ax2.set_title('Model FC')
        plt.savefig(run.joinpath('figures', 'fc_heatmap.png'))
        plt.close()
        run.joinpath('connectomes').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(real_fc, index=labels, columns=labels).to_csv(run.joinpath('connectomes', 'real.csv'))
        pd.DataFrame(model_fc, index=labels, columns=labels).to_csv(run.joinpath('connectomes', 'model.csv'))
    
    df.index.name = 'Scan'
    df.to_csv(f'{work_dir.name}.csv')
    print(df['r'].mean(), df['r'].median(), df['r'].std())
    print(df['recon_err'].mean(), df['recon_err'].median(), df['recon_err'].std())