import torch
import numpy as np
from tqdm import tqdm

normwindows=3 
windowlength=90
polyorder=4
dataset = torch.load(f"preprocessed_spectra.pt", weights_only = False)

all_time = []
all_descriptors = []
all_fluxes = []

for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    
    
    
    all_time.append(sample['time'])                 # scalar
    all_descriptors.append(sample['descriptor'].numpy())   # shape: [9]
    all_fluxes.append(sample['flux'])
    


time_all = np.array(all_time)           # [num_samples]
desc_all = np.stack(all_descriptors)    # [num_samples, 9]
fluxes_all = np.stack(all_fluxes)
print("fluxes: "+str(fluxes_all))
print("fluxes nan columns:", np.any(np.isnan(fluxes_all), axis=0))

# Optional: renormalize descriptors by some absolute vector if needed
norm_vector = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0E33, 1.0E33, 1.0E33])
coef_norm_const = 1.0E35


desc_all = desc_all / norm_vector

print("Any inf or nan in descriptor columns?")
print("Inf columns:", np.any(np.isinf(desc_all), axis=0))
print("Nan columns:", np.any(np.isnan(desc_all), axis=0))

print("Indices with inf in desc_all:")
print(np.where(np.isinf(desc_all)))

print("Sample values of bad elements:")
idxs_inf = np.where(np.isinf(desc_all))
for i, j in zip(*idxs_inf):
    print(f"Sample {i}, col {j}, value: {desc_all[i, j]}, descriptor: {desc_all[i]}")

#compute stats for fluxes
fluxes_all /= coef_norm_const
fluxes_mean = np.mean(fluxes_all)
fluxes_std = np.std(fluxes_all) 

# Compose normalization statistics dict
norm_stats = {
    'time_mean': float(np.mean(time_all)),
    'time_std': float(np.std(time_all)),
    'descriptor_mean': torch.tensor(np.mean(desc_all, axis=0)) * norm_vector,
    'descriptor_std': torch.tensor(np.std(desc_all, axis=0)) * norm_vector,
    'fluxes_mean': fluxes_mean*coef_norm_const,
    'fluxes_std': fluxes_std*coef_norm_const
}

print("norm_stats:", norm_stats)

# Save the normalization dictionary
torch.save(norm_stats, f'normalization_stats.pt')
print(f"Saved updated normalization stats to 'normalization_stats.pt'.")