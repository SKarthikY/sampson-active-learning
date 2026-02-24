import torch 
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from helpers import *
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import glob
from simple_MLP import *

def train(d_model=128, nhead=8, num_layers=4, 
          learnedPE=True, lr = 2.5e-4, 
          weight_decay = 0.01,
          batch_size = 64,
          epochs = 100, 
          normalization_stats_file = "NeuralNetworks/NN_grid/active_learning/normalization_stats.pt", 
          preprocessed_spectra_file = "/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/preprocessed_spectra.pt",
          supplemental_spectra_file = "/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/supp_preprocessed_spectra.pt",
          idx = 0, MLP = True
          ):
    
    run = None
    
    wandb.login(key=os.environ.get("WANDB_API_KEY"))


    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="karthik_yadavalli-harvard-university",
        # Set the wandb project where this run will be logged.
        project="NN_grid_training-cannon",
        # Track hyperparameters and run metadata.
        config={
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "learnedPE": learnedPE,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "idx": idx,
        },
    )
    
    model_name = f"dim{d_model}_nhead{nhead}_numlayers{num_layers}_learnedPE{learnedPE}_lr{lr}_weightdecay{weight_decay}_batchsize{batch_size}_epochs{epochs}_idx{idx}_MLP{MLP}"
    print(model_name)
    print("preprocessed_spectra_file: "+str(preprocessed_spectra_file))
    print("normalization_stats_file: "+str(normalization_stats_file))
    mkdir('../model_ckpt/')
    mkdir('../model_ckpt/'+model_name+"/")
    out_dir = '../model_ckpt/'+model_name+"/"
    
    
    
    orig_data = torch.load(preprocessed_spectra_file, weights_only = False)
    supp_data = torch.load(supplemental_spectra_file, weights_only=False)


    all_sample_ids = sorted({entry['sample_id'] for entry in orig_data})
    train_sample_ids, test_sample_ids = train_test_split(
        all_sample_ids, train_size=0.8, random_state=42
    )

    train_data = [entry for entry in orig_data if entry['sample_id'] in train_sample_ids]
    test_data  = [entry for entry in orig_data if entry['sample_id'] in test_sample_ids]
    train_data.extend(supp_data) #add supplemental grid as training data only

    mean_std_dict = torch.load(normalization_stats_file, weights_only = False)
    transform = NormalizeSpectralData(mean_std_dict)
    train_dataset = FastSupernovaDataset(samples=train_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    #### models ####

    #check if a model exists already:
    saved_models = glob.glob(out_dir + "/*.pth")
    if len(saved_models) > 0:
        print("Found existing model(s) in "+out_dir+", loading the latest one.")
        if os.path.exists(out_dir + "/final.pth"):
            print("Model already done and trained. Exiting.")
            run.finish()
            return
        latest_model_file = sorted(saved_models)[-1]
        print("Loading model: "+str(latest_model_file))
        model = torch.load(latest_model_file, map_location="cpu", weights_only = False)
    else:

        
        if MLP:
            model = SimpleFluxMLP(n_physical_param=10, n_wavelength=602, d_model=d_model, num_layers=num_layers)
        else:
            model = FluxTransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers, learnedPE=learnedPE)


    '''if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)'''
    

    
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
         if torch.cuda.is_available() else []
    is_mig = any("MIG" in n for n in names)

    if torch.cuda.device_count() > 1 and not is_mig:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"Using single device: {device} (MIG or single-GPU setup)")


    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    
    criterion = nn.MSELoss() 
    
    model, losses = train_model(model, train_loader, 
                criterion, optimizer, scheduler, out_dir = out_dir,
                epochs = epochs, 
                device = device,
                wandb_run = run
                )
    
    
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save, out_dir + "/final.pth")

    
    plt.plot(np.log10(np.array(losses)))
    plt.xlabel("Epoch")
    plt.ylabel("logLoss")
    plt.tight_layout()
    plt.savefig(f"../experimental_results/"+model_name+"/losses.pdf", bbox_inches='tight')
    plt.close()

    run.finish()



import fire 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Visible CUDA devices: {num_devices}")
        for i in range(num_devices):
            name = torch.cuda.get_device_name(i)
            print(f"CUDA device {i}: {name}")
    else:
        print("No CUDA devices available")

    partition = os.environ.get("SLURM_JOB_PARTITION", "UNKNOWN")
    print(f"\nRunning on SLURM partition: {partition}\n\n==========================================================================================\n")

    fire.Fire(train)
    