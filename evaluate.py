import torch
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

torch.manual_seed(42)
np.random.seed(42)

def remove_diag(arr):

    return arr[~np.eye(arr.shape[0], dtype=bool)]


def sort_fn(path: Path):
    loss = path.name.split('_')[-1].removesuffix('.pt')
    return float(loss)


def find_best_loss(path):
    models = [x for x in path.rglob('**/*best_val_loss*.pt')]
    return sorted(models, key=sort_fn)[0]


def find_raw_data(path, name):
    return list(path.rglob(f'**/{name}*timeseries.csv'))[0]


def get_model_fc(model, steps, sim_data_length, num_regions):
    
    device = next(model.parameters()).device
    model.eval()
    outputs = [torch.zeros(num_regions, device=device, dtype=torch.float).to(device) for _ in range(steps)]
    for _ in range(sim_data_length):
        noise = 0.1 * np.random.randn(steps, num_regions)
        sim_input = torch.vstack(outputs[-steps:]).to(device) + torch.tensor(noise, dtype=torch.float).to(device)
        outputs.append(
            model(sim_input.unsqueeze(0).to(device))
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
    work_dir = Path('npi_result_fine-tune_test/')
    
    with open('region_labels.txt', 'r') as f:
        labels = f.readlines()
    labels = [x.removesuffix('\n') for x in labels]
    
    r_p = pd.DataFrame(columns=['r', 'p'])
    neworderlr = np.loadtxt('/data5/projects/bwood/NPI/NPI/Whole-brain_EC/MNI-MMP1/neworderlr.txt')
    neworder = np.array([np.where(neworderlr == i)[0][0] for i in range(len(neworderlr))])
    for run in tqdm(work_dir.iterdir(), total=len(list(work_dir.iterdir()))):
        cfg = OmegaConf.load(run.joinpath('config.yaml'))
        model = get_model(
            cfg.model.name, 
            **cfg.model.kwargs 
        )
        model.load_state_dict(
            torch.load(find_best_loss(run))['model_state'],
        )
        model.eval()
        model.to('cuda:0')
        model_fc = get_model_fc(model, cfg.data.train.step, 1200, 360)
        
        # bold_data = pd.read_csv(find_raw_data(Path('data_like-npi/hcp'), 'sub-100206_ses-3T_task-rest_acq-rl_space-MNIICBM152_desc-preproc'), index_col=0).to_numpy().astype(np.float64)
        print(cfg.run_name)
        bold_data = pd.read_csv(find_raw_data(Path('data_like-npi/hcp'), cfg.run_name), index_col=0).to_numpy().astype(np.float64)
        real_fc = ConnectivityMeasure(kind='correlation').fit_transform(bold_data[30:, :360])[0]
        
        r, p = pearsonr(remove_diag(real_fc), remove_diag(model_fc))
        r_p.at[run.name, 'r'] = r
        r_p.at[run.name, 'p'] = p
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
    
    r_p.index.name = 'Scan'
    r_p.to_csv('rs_ps_dev.csv')