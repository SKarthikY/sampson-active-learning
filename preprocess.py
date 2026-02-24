import os
import torch
import numpy as np
import pandas as pd
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import astropy.units as u
from astropy import constants as const
from scipy.signal import *
from sedoNNa.utils import *

# Physical constants (reuse your definitions)
c = 3.0E8 * u.meter/u.second
angstrom = 1.0E-10 * u.meter
erg = 1.0E-7 * u.kilogram*(u.meter/u.second)**2
cm = 1.0E-2 * u.meter
hz = 1/u.second
freq_flux = erg/(cm**2*u.second*hz)
wav_flux = erg/(cm**2*angstrom*u.second)
sec_per_day = 86400

# ====== Parameters ======
orig_dir = '/n/netscratch/avillar_lab/Everyone/karthik/aCOperation/models_4d_0.2dt/'
supplemented_dir = '/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/supplemental_grid/'
N_wavelengths = 602
wav_min = 2000
wav_max = 10000
normwindows=3
windowlength=90
polyorder=4
output_file = "preprocessed_spectra.pt"
def get_num_workers():
    # Most common: --cpus-per-task
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    # Sometimes only this is set
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    # Generic fallback
    return os.cpu_count() or 1
num_workers = get_num_workers()
# ========================

fixed_wav_grid = np.linspace(wav_min+200, wav_max-300, N_wavelengths)
all_sample_ids = [x for x in os.listdir(orig_dir) if os.path.isdir(os.path.join(orig_dir, x))]
supplemental_sample_ids = [x for x in os.listdir(supplemented_dir) if os.path.isdir(os.path.join(supplemented_dir, x))]


def get_spectrum_from_file(file, time):
    #read in the spectrum for a given time
    dat = np.loadtxt(file)
    times = (dat[:,0] * u.second).to(u.day).value
    freq = (dat[:,1]*(1/u.second))
    wav = (const.c/freq).to(u.angstrom).value
    ff = dat[:,2]*u.erg/(u.cm**2)
    flux = ((ff*freq**2/const.c).to(u.erg/(u.cm**2*u.angstrom*u.second))).value


    idx = np.where(times == time)
    wav = wav[idx]
    flux = flux[idx]

    idx = np.where((wav > 2000)&(wav < 10000))
    wav = wav[idx]
    flux = flux[idx]

    return wav, flux


#get a function that can calculate the total flux in a given filter. The filter profiles directory shuold also be passed in.
def get_mags_from_spec(wav, spec, filter_prof_dir = '/n/home07/kyadavalli/scratch/NeuralNetworks/for_github/filter_profs/'):
    mags = []
    filts = []
    filters = {'u': 'SLOAN_SDSS.u', 'g': 'SLOAN_SDSS.g', 'r': 'SLOAN_SDSS.r', 'i': 'SLOAN_SDSS.i', 'z': 'SLOAN_SDSS.z'}
    for fil in filters:
        #read in the bandpass in this filter
        filter_data = np.loadtxt(filter_prof_dir+"/"+str(filters[fil])+".dat")
        filter_wavs = filter_data[:, 0]
        filter_transmission = filter_data[:, 1]
        
        #calculate the photometric point in this filter
        #spectra from sedona are outputted as L_nu, right? So total luminosity at a given \nu
        #I need to divide it by 4*pi*(10 parsec)**2 so it is flux, rather than luminosity
        d = (10*u.parsec).to(u.cm).value
        denom = 4*np.pi*d**2.0
        mag = compute_photometry(wav, spec/denom, filter_wavs, filter_transmission)
        filts.append(fil)
        mags.append(mag)
    return np.asarray(mags)
    

def smoothen(wav, flux, norm_windows=normwindows, window_length=windowlength, polyorder=polyorder):
    # 1. Smooth the flux
    smoothed_flux = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    
    # 2. Prepare output array
    corrected_flux = smoothed_flux.copy()
    
    # 3. Find window edges (ensure roughly equal numbers of points per window)
    indices = np.linspace(0, len(wav), norm_windows + 1, dtype=int)
    
    # 4. Loop over windows
    for i in range(norm_windows):
        i0, i1 = indices[i], indices[i+1]
        # Integrate original and smoothed spectra over this window
        orig_int = np.trapz(flux[i0:i1], wav[i0:i1])
        smth_int = np.trapz(smoothed_flux[i0:i1], wav[i0:i1])
        scale = 1.0
        if smth_int != 0:
            scale = orig_int / smth_int
        # Apply the scaling within this window
        corrected_flux[i0:i1] *= scale
    
    return smoothed_flux, corrected_flux


def process_sample(sample_id, data_dir):
    sample_path = os.path.join(data_dir, sample_id)
    if "supp" in data_dir:
        sample_id = "supp_"+sample_id
    spec_path = os.path.join(sample_path, 'spectrum_final.dat')
    sample_txt = os.path.join(sample_path, 'sample.txt')
    if not os.path.exists(spec_path) or not os.path.exists(sample_txt):
        return []  # <-- Return empty list if sample missing
    
    # Descriptor
    sep = ',' if 'Callie' not in sample_path else ' '
    descriptor_df = pd.read_csv(sample_txt, sep=sep, header=0)
    descriptor = torch.tensor([
        descriptor_df['D'].values[0],
        descriptor_df['R_2'].values[0],
        descriptor_df['R_28'].values[0],
        descriptor_df['R_opacity'].values[0],
        descriptor_df['min_vel'].values[0],
        descriptor_df['max_vel'].values[0],
        descriptor_df['total_2'].values[0],
        descriptor_df['total_28'].values[0],
        descriptor_df['total_opacity'].values[0],
    ], dtype=torch.float32)

    data = pd.read_csv(spec_path, sep=r'\s+', header=None, names=['time', 'frequency', 'flux', 'fluxerr'], comment='#')
    frequency = data['frequency'].values * hz
    flux_fnu = data['flux'].values * freq_flux
    wav = (c / frequency)
    wav_cm = wav.to(cm)
    flux_flambda = (flux_fnu * c / wav_cm ** 2).to(wav_flux)
    wav_angstrom = wav.to(angstrom).value

    mask = (wav_angstrom > wav_min) & (wav_angstrom < wav_max) & np.isfinite(flux_flambda.value)
    wav = wav_angstrom[mask]
    flux = flux_flambda.value[mask]
    time = data['time'].values[mask]

    mask = (wav > wav_min) & (wav < wav_max)
    wav = wav[mask]
    flux = flux[mask]
    time = time[mask]

    mask_time = (time > 7*sec_per_day) & (time < 40*sec_per_day)
    wav = wav[mask_time]
    flux = flux[mask_time]
    time = time[mask_time]

    this_sample_results = []
    for t in np.unique(time):
        this_mask = (time == t)
        wav_t = wav[this_mask]
        flux_t = flux[this_mask]
        wav_t = wav_t[::-1]
        flux_t = flux_t[::-1].copy()
        if len(wav_t) < 10:
            continue
        flux_interp = np.interp(fixed_wav_grid, wav_t, flux_t, left=0, right=0)

        
        #first check if there are any bands where absolute magnitude is dimmer than -10. If yes, don't store this spectrum.
        mags = get_mags_from_spec(fixed_wav_grid, flux_interp)
        if np.max(mags) > -10:
            continue

            
        #smoothen these spectra
        smoothed_flux, _ = smoothen(fixed_wav_grid, flux_interp)

        min_flux = np.min(smoothed_flux[np.where(smoothed_flux > 0)])
        smoothed_flux[np.where(smoothed_flux <= 0)] = min_flux

        log_flux = np.log10(smoothed_flux)
        

        this_sample_results.append({
            'descriptor': descriptor.clone(),
            'time': float(t),
            'wav': fixed_wav_grid.copy(),
            'flux': log_flux.copy(),
            'sample_id': sample_id
        })
    return this_sample_results

def main():
    all_results = []

    #running over original grid samples first
    print(f"Processing {len(all_sample_ids)} sample objects with {num_workers} workers")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_sampleid = {executor.submit(process_sample, sid, orig_dir): sid for sid in all_sample_ids}
        for i, future in enumerate(as_completed(future_to_sampleid)):
            sid = future_to_sampleid[future]
            try:
                result = future.result()
                all_results.extend(result)
                if (i+1) % 10 == 0:
                    print(f"Finished {i+1} samples, {len(all_results)} spectra.")
            except Exception as e:
                print(f"Error processing {sid}: {e}")

    print(f"Saving {len(all_results)} spectra from original grid to {output_file}")
    torch.save(all_results, output_file)
    
    
    #running over supplemented grid samples
    all_results = []  # reset results to only store supplemented samples
    print(f"Processing {len(supplemental_sample_ids)} supplemental sample objects with {num_workers} workers")
    supplemental_output_file = "supp_"+output_file
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_sampleid = {executor.submit(process_sample, sid, supplemented_dir): sid for sid in supplemental_sample_ids}
        for i, future in enumerate(as_completed(future_to_sampleid)):
            sid = future_to_sampleid[future]
            try:
                result = future.result()
                all_results.extend(result)
                if (i+1) % 10 == 0:
                    print(f"Finished {i+1} supplemental samples, {len(all_results)} spectra.")
            except Exception as e:
                print(f"Error processing supplemental sample {sid}: {e}")

    print(f"Saving {len(all_results)} spectra to {supplemental_output_file}")
    torch.save(all_results, supplemental_output_file)
    print("Done")

if __name__ == '__main__':
    main()