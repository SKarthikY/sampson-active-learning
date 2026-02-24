import pandas as pd
import torch
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from sedoNNa.utils import *
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from helpers import *
from tqdm import tqdm
import matplotlib.ticker as mtick
from utils import *


eval_model = False
create_new_jobs = False
recalculate_normalization = False
train_new_em = True

to_run = 30
ran = 0


if not create_new_jobs:
    to_run = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = '/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/model_ckpt'
norm_stats_file='/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/normalization_stats.pt'
preprocessed_spectra_file='/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/preprocessed_spectra.pt'

mean_std_dict = torch.load(norm_stats_file, weights_only=False)
fluxes_mean = mean_std_dict['fluxes_mean']
fluxes_std = mean_std_dict['fluxes_std']
time_mean = mean_std_dict['time_mean']
time_std = mean_std_dict['time_std']
descriptor_mean = mean_std_dict['descriptor_mean'].float().to(device)
descriptor_std = mean_std_dict['descriptor_std'].float().to(device)


N = 3
M = 10
model = "dim512_nhead64_numlayers7_learnedPETrue_lr0.008_weightdecay0.07_batchsize512_epochs1000_idx0_MLPTrue"
epochs = 200

#1. read in the model, run it over the entire test set, and find the delta mag error for each sample in the test set.
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



if eval_model:
    evaluate_model(ckpt_dir, model, epochs, sample_ids, wav)

#2. sort the test set by delta mag error, and select the top N samples with the highest error.
#read in delta_mag_error file
to_read = "lc_experimental_results/"+str(model)+"/"+str(epochs)+"/lc_properties.txt"
lc_prop_dict = read_file(to_read)['dict']
sids = np.asarray(lc_prop_dict[0], dtype = float)
avg_mag_diff = np.asarray(lc_prop_dict[1], dtype = float)
max_mag_diff = np.asarray(lc_prop_dict[2], dtype = float)   

#sort in descending order by max_mag_diff
sorted_indices = np.argsort(avg_mag_diff)[::-1]
sids = sids[sorted_indices]
avg_mag_diff = avg_mag_diff[sorted_indices]
max_mag_diff = max_mag_diff[sorted_indices]

def get_sample_location(sid):
    sid = str(int(sid))
    #read in sample.txt for this sample id, and return the physical parameters as a tensor
    sample_txt_path = f"/n/home07/kyadavalli/scratch/aCOperation/models_4d_0.2dt/{sid}/sample.txt"
    if not os.path.exists(sample_txt_path):
        print("sample txt not found for sample id: "+str(sid))
        return None
    sample_df = pd.read_csv(sample_txt_path, sep=',')
    sample_loc = torch.tensor([
        sample_df['D'].values[0],
        sample_df['R_2'].values[0],
        sample_df['R_28'].values[0],
        sample_df['R_opacity'].values[0],
        sample_df['min_vel'].values[0],
        sample_df['max_vel'].values[0],
        sample_df['total_2'].values[0],
        sample_df['total_28'].values[0],
        sample_df['total_opacity'].values[0],
    ]).to(device).float()

    #normalize this sample location using the descriptor mean and std
    return (sample_loc - descriptor_mean) / descriptor_std


new_sample_points = {"D": [], "R_2": [], "R_28": [], "R_opacity": [], "min_vel": [], "max_vel": [], "total_2": [], "total_28": [], "total_opacity": []}
d = 0.1

#2a. loop through each of the N samples. for each sample i in N samples, sample M distances from a half gaussian with standard deviation d.
for i in range(N):
    sid = int(sids[i])
    next_sample = int(sids[i+1])
    print("Working on sample id: "+str(sid))
    for j in range(M):
        #2b. for each distance sampling l from M distances, step distance l away from sample i in the direction of the sample with the next highest error. This is the new sample point.
        l = np.abs(np.random.normal(loc=0.0, scale=d))


        #step distance l away from the sample in the direction of the sample with the next highest error. This is the new sample point.

        sample_loc = get_sample_location(sid)  # get the physical parameters for this sample
        next_sample_loc = get_sample_location(next_sample)  # get the physical parameters for the next sample
        if sample_loc is None or next_sample_loc is None:
            print("Could not get sample location for sample id: "+str(sid)+" or "+str(next_sample)+". Skipping...")
            continue
        direction = next_sample_loc - sample_loc
        direction = direction / torch.norm(direction)  # normalize the direction vector
        new_sample_loc = sample_loc + l * direction  # step distance l in the direction of the next sample

        new_sample_loc = new_sample_loc * descriptor_std + descriptor_mean  # unnormalize the new sample location

        #2c. add this new point to a list of points where sedona will be run
        new_sample_points["D"].append(new_sample_loc[0].item())
        new_sample_points["R_2"].append(new_sample_loc[1].item())
        new_sample_points["R_28"].append(new_sample_loc[2].item())
        new_sample_points["R_opacity"].append(new_sample_loc[3].item())
        new_sample_points["min_vel"].append(new_sample_loc[4].item())
        new_sample_points["max_vel"].append(new_sample_loc[5].item())
        new_sample_points["total_2"].append(new_sample_loc[6].item())
        new_sample_points["total_28"].append(new_sample_loc[7].item())
        new_sample_points["total_opacity"].append(new_sample_loc[8].item())

        







#3 Loop through the N*M new sample points. Generate a sedona batch file for each point, run sedona, and add the new data to the training set.



mkdir("supplemental_grid/")
files = glob.glob("supplemental_grid/*/mod.mod")
max_existing_sample = np.max([int(f.split("/")[-2]) for f in files]) if len(files) > 0 else -1
for i in range(len(new_sample_points["D"])):
    if ran < to_run:
        out_dir = "supplemental_grid/"+str(max_existing_sample+1+i)+"/"
        D = new_sample_points["D"][i]
        R_2 = new_sample_points["R_2"][i]
        R_28 = new_sample_points["R_28"][i]
        R_opacity = new_sample_points["R_opacity"][i]
        min_vel = new_sample_points["min_vel"][i]
        max_vel = new_sample_points["max_vel"][i]
        total_2 = new_sample_points["total_2"][i]
        total_28 = new_sample_points["total_28"][i]
        total_opacity = new_sample_points["total_opacity"][i]
        
        construct_run(out_dir, D, R_2, R_28, R_opacity, min_vel, max_vel, total_2, total_28, total_opacity, start_time = 5.0, stop_time = 50, dt = 0.2, hours = 3, mass_dict = None, vel = None, job_name = 'supp_'+str(max_existing_sample+1+i))

        write_to_file(out_dir+"info.txt", "This is supplemental sample "+str(i%M)+" for original sample "+str(int(sids[math.floor(i/M)])))
    
    
        print("Starting job for sample id: "+str(max_existing_sample+1+i))
        os.system("cd "+out_dir+"; sbatch run_batch.sub")
        ran += 1



#4 Once all new sedona runs are done, recalculate the normalizations and preprocessed spectra .pt files for the training set, and retrain the emulator model on the new training set

if recalculate_normalization:
    os.system("sbatch run1_batch.sub")


#retrain emulator on the new .pt files in this directory
#generate a run_small.sh type of file, borrowing from generate_sbatch.py, that will run the train_small.py script on the new .pt files in this directory, and submit this file to the cluster.


def get_sbatch_code(idx, d_model, nhead, num_layers, lr, weight_decay, batch_size, MLP = True):
    tor = """#!/bin/sh
#SBATCH --cpus-per-task 2
#SBATCH --nodes 1
#SBATCH -t 00-00:15:00
#SBATCH -p itc_gpu,gpu,gpu_h200,gpu_requeue,itc_gpu_requeue
#SBATCH --gres=gpu:3
#SBATCH --constraint="a100|h100|h200"
#SBATCH --mem=4G
#SBATCH --output=logs/"""+str(idx)+""".out
#SBATCH --error=logs/"""+str(idx)+""".err
#SBATCH --job-name=grid_"""+str(idx)+"""

source ~/.bashrc
module load gcc/12.2.0-fasrc01
module load mpich/4.1-fasrc01
module load gsl/2.7-fasrc01
module load hdf5/1.14.0-fasrc01
module load python/3.10.9-fasrc01
mamba activate sedona_venv

## pip install -e /n/home07/kyadavalli/scratch/NeuralNetworks/for_github/package/
## pip install fire
## pip install -e /n/home07/kyadavalli/Astro_Code/helpers/helpers/
## pip install scikit-learn

normalization_stats_file=/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/normalization_stats.pt
preprocessed_spectra_file=/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/preprocessed_spectra.pt
supp_preprocessed_spectra_file=/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/supp_preprocessed_spectra.pt
train_small_file=/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/train_small.py

python $train_small_file --d_model """+str(d_model)+""" --nhead """+str(nhead)+""" --num_layers """+str(num_layers)+""" --learnedPE True --lr """+str(lr)+""" --weight_decay """+str(weight_decay)+""" --batch_size """+str(batch_size)+""" --epochs 300 --normalization_stats_file $normalization_stats_file --preprocessed_spectra_file $preprocessed_spectra_file --idx """+str(idx)+""" --supplemental_spectra_file $supp_preprocessed_spectra_file"""

    job_signature = f"{idx}_dim{d_model}_nhead{nhead}_numlayers{num_layers}_learnedPETrue_lr{lr}_weightdecay{weight_decay}_batchsize{batch_size}"
    return tor, job_signature


if train_new_em:
    dir1 = "training_emulator/"
    trained_ems = glob.glob(dir1+"run_em_*.sh")
    idx = '0'
    if len(trained_ems) > 0:
        idx = str(1+np.max([int(i.split('.')[0].split("_")[-1]) for i in trained_ems]))



    ss, job_signature = get_sbatch_code(idx, d_model=512, nhead=64, num_layers=7, lr=0.008, weight_decay=0.07, batch_size=512)


    mkdir(dir1)
    write_to_file(dir1+"run_em_"+idx+".sh", ss)
    os.system('cd '+dir1+'; sbatch run_em_'+idx+'.sh; cd ..')

#5 Evaluate the new emulator on the test set, and repeat steps 1-5 until the emulator performance converges.