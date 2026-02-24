#this script just simply runs the emulator on one sample in the supplemental grid or in the original grid. 
#It should provice multiband light curves and spectra for the entire time series.

from sedoNNa.utils import *
import pandas as pd
from utils import *
import torch
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from sedoNNa.utils import *
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#prescribe which model to work with
ckpt_dir = '/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/model_ckpt'
model = "dim512_nhead64_numlayers7_learnedPETrue_lr0.008_weightdecay0.07_batchsize512_epochs1000_idx0_MLPTrue"


norm_stats_file='/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/normalization_stats.pt'
preprocessed_spectra_file='/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/preprocessed_spectra.pt'



mean_std_dict = torch.load(norm_stats_file, weights_only=False)
fluxes_mean = mean_std_dict['fluxes_mean']
fluxes_std = mean_std_dict['fluxes_std']
time_mean = mean_std_dict['time_mean']
time_std = mean_std_dict['time_std']
descriptor_mean = mean_std_dict['descriptor_mean'].float().to(device)
descriptor_std = mean_std_dict['descriptor_std'].float().to(device)

all_data = torch.load(preprocessed_spectra_file, weights_only=False)
all_sample_ids = sorted({entry['sample_id'] for entry in all_data})
train_sample_ids, test_sample_ids = train_test_split(
    all_sample_ids, train_size=0.8, random_state=42
)
sample_ids = set(test_sample_ids)
data = [entry for entry in all_data if entry['sample_id'] in sample_ids ]
transform = NormalizeSpectralData(mean_std_dict)
dataset = FastSupernovaDataset(samples=data, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True,  drop_last=True, num_workers=4, pin_memory=True)
wav = next(iter(loader))['wav'][0].to(device)


sample_ids = ["628", "supp_0", "supp_1", "supp_2", "supp_3", "supp_4", "supp_5", "supp_6", "supp_7", "supp_8", "supp_9", "supp_10", 'supp_11', 'supp_12', 'supp_13']
mkdir("testing_emulator_results/")
mkdir(f"testing_emulator_results/{model}/")


for epochs in ['final']:
    mkdir(f"testing_emulator_results/{model}/{epochs}/")
    evaluate_model(ckpt_dir, model, epochs, sample_ids, wav, save_dir = f"testing_emulator_results/{model}/{epochs}/")