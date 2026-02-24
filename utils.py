import glob
import numpy as np
from helpers import *
import torch
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from sedoNNa.utils import *
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from tqdm import tqdm

def get_lc_properties(LC_file, filt = 'Lbol(erg/s)'):
    lc, times = get_lc_from_file(LC_file, filt = filt)
    if lc is None:
        return None, None, None, None
    
    if 'SDSS' in filt:
        lc = (10**(-1.0*lc/2.5) * 3631*jansky * 4*np.pi*(10*u.pc)**2).decompose().to(u.erg/(u.second*hz)).value
        
    peak_lum = np.max(lc)
    peak_time = times[find_nearest(lc, peak_lum)]
    rise_time = get_rise_time(times, lc)
    fall_time = get_fall_time(times, lc)
    
    return peak_lum, peak_time, rise_time, fall_time

d = (10 * u.parsec).to(u.cm).value
denom = 4 * np.pi * d**2.0


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

filter_prof_dir = '/n/home07/kyadavalli/scratch/NeuralNetworks/for_github/filter_profs/'
filters = {
    'u': 'SLOAN_SDSS.u',
    'g': 'SLOAN_SDSS.g',
    'r': 'SLOAN_SDSS.r',
    'i': 'SLOAN_SDSS.i',
    'z': 'SLOAN_SDSS.z',
}
filter_tensors = {}
for fil, fname in filters.items():
    filter_data = np.loadtxt(f"{filter_prof_dir}/{fname}.dat")
    filter_wavs = filter_data[:, 0]
    filter_transmission = filter_data[:, 1]
    fw_t = torch.as_tensor(filter_wavs, dtype=torch.float64, device=device)
    ft_t = torch.as_tensor(filter_transmission, dtype=torch.float64, device=device)
    filter_tensors[fil] = (fw_t, ft_t)

bbox_props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=0.5, alpha=1.0)

def evaluate_model(ckpt_dir, ckpt, epochs, sample_ids, wav, save_dir = None):
    
    model = torch.load(f"{ckpt_dir}/{ckpt}/{epochs}.pth", map_location = device, weights_only=False)
    model.eval()
    criterion = nn.MSELoss()

    mkdir("experimental_results")
    mkdir("experimental_results/"+str(ckpt))
    mkdir("experimental_results/"+str(ckpt)+"/"+str(epochs))

    print("Working on ckpt: "+str(ckpt)+", epoch: "+str(epochs))
    
    if save_dir is None:
        mkdir("lc_experimental_results")
        mkdir("lc_experimental_results/"+str(ckpt))
        mkdir("lc_experimental_results/"+str(ckpt)+"/"+str(epochs))
        save_dir = "lc_experimental_results/"+str(ckpt)+"/"+str(epochs)+"/"

    mkdir(save_dir)
    lc_properties_file = save_dir+"/lc_properties.txt"

    if os.path.exists(lc_properties_file) and os.path.exists("lc_experimental_results/"+str(ckpt)+"/"+str(epochs)+"/figs/max_ft.pdf"):
        print("lc properties file already exists for ckpt: "+str(ckpt)+", epoch: "+str(epochs)+". Skipping...")
        return

    write_to_file(lc_properties_file, "# Sample\tAvg_Mag_Diff\tMax_Mag_Diff\tPeak_Lum\tPeak_Time\tRise_Time\tFall_Time\n", append = False)
    for sid in tqdm(sample_ids):

        out_file = save_dir+"/"+str(sid)+"/lc.txt"
        mkdir(save_dir+f"{sid}/")

        tow = "# Time (Days)\tLbol (erg/s)\tMbol"
        for fil, (fw_t, ft_t) in filter_tensors.items():
            tow += f"\tSDSS_{fil}"
        tow += "\n"
        write_to_file(out_file, tow, append = False)
        
        
        sample_txt_path = f"/n/home07/kyadavalli/scratch/aCOperation/models_4d_0.2dt/{sid}/sample.txt"
        if "supp" in sid:
            sample_txt_path = f"/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/supplemental_grid/{sid[5:]}/sample.txt"
        if not os.path.exists(sample_txt_path):
            print("sample txt not found for sample id: "+str(sid))
            continue
        sample_df = pd.read_csv(sample_txt_path, sep=',')


        # Pass in physical parameters from this sample to emulator
        with torch.no_grad():
            param_tensor = torch.tensor([
                sample_df['D'].values[0],
                sample_df['R_2'].values[0],
                sample_df['R_28'].values[0],
                sample_df['R_opacity'].values[0],
                sample_df['min_vel'].values[0],
                sample_df['max_vel'].values[0],
                sample_df['total_2'].values[0],
                sample_df['total_28'].values[0],
                sample_df['total_opacity'].values[0]
            ], dtype=torch.float32, device=device).unsqueeze(0)

            param_tensor = (param_tensor - descriptor_mean) / descriptor_std


            for t in np.arange(7.5*86400, 40.5*86400, 1.0*86400):
                time_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
                time_tensor = (time_tensor - time_mean) / time_std
                input_tensor = torch.cat((param_tensor, time_tensor), dim=1)

                output = model(input_tensor).to(torch.float64)
                spec_pred = (10.0**(output * fluxes_std + fluxes_mean))
                
                lbol = torch.trapz(spec_pred.squeeze(), wav)
                mbol = 4.74-2.5*torch.log10(lbol/3.83e33)
                spec_pred /= denom
                
                #print("predicted spectrum: "+str(spec_pred))
                #take a full convolution to get the bolometric luminosity

                tow = f"{t/86400}\t{lbol}\t{mbol}"
                #loop through the filters, and convolve the predicted spectrum with the filter to get the predicted magnitude in that filter. Then compute the true magnitude in that filter using the true spectrum, and compute the del_mag
                for fil, (fw_t, ft_t) in filter_tensors.items():

                    pred_mag_batch = compute_photometry_torch(wav, spec_pred, fw_t, ft_t)  # (batch,)
                    
                    #write this to file
                    tow += f"\t{pred_mag_batch.item()}"
                tow += "\n"
                write_to_file(out_file, tow, append = True)

        #read in light curve from sedona
        dir1 = '/n/home07/kyadavalli/scratch/aCOperation/models_4d_0.2dt/'
        if "supp" in sid:
            dir1 = '/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/active_learning/supplemental_grid/'

        filt_list = ['g', 'r', 'i', 'z']

        # Create one big figure with a 2x2 grid of subplots
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # adjust size as needed
        axes = np.array(axes).reshape(2, 2)            # ensure 2x2 indexing

        for idx, filt in enumerate(filt_list):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            if 'supp' in sid:
                sed_lc, sed_times = get_lc_from_file(dir1+""+str(sid)[5:]+"/lightcurve.out", filt = "SDSS_"+filt)
            else:
                sed_lc, sed_times = get_lc_from_file(dir1+""+str(sid)+"/lightcurve.out", filt = "SDSS_"+filt)
            if sed_lc is None:
                print("Light curve not found for sample id: "+str(sid))
                continue
            
            lc, times = get_lc_from_file(out_file, filt = "SDSS_"+filt)
            if lc is None:
                print("Light curve not found for sample id: "+str(sid))
                continue

            idx = np.where((times > 7) & (times < 40))
            sed_idx = np.where((sed_times > 7) & (sed_times < 40))
            avg_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).mean(), 2))
            max_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).max(), 2))

            ax.plot(
                sed_times, sed_lc,
                label=f"True {filt} LC",
                color=sKy_colors['blue'],
                alpha=0.7,
                linewidth=4,
            )
            ax.plot(
                times, lc,
                label=f"Predicted {filt} LC",
                color=sKy_colors['green'],
                alpha=0.7,
                linewidth=4,
            )

            if row == 0:
                ax.set_xlabel("")
                ax.set_xticks([])

            else:
                ax.set_xlabel("Time (Days)", fontsize=18)
                
            ax.set_ylabel("", fontsize=18)

            ul = np.min([np.min(sed_lc), np.min(lc)])
            ax.set_ylim([ul + 5, ul - 0.3])  # same convention as before

            ax.set_title(f"{filt}-band, Avg|Max Mag Diff: "+str(avg_mag_diff)+" | "+str(max_mag_diff), fontsize=25)
            ax.legend(fontsize=12)


            if filt == 'r':
                if 'supp' in sid:
                    peak_lum, peak_time, rise_time, fall_time = get_lc_properties(dir1+""+str(sid)[5:]+"/lightcurve.out", filt = 'SDSS_r')
                else:
                    peak_lum, peak_time, rise_time, fall_time = get_lc_properties(dir1+""+str(sid)+"/lightcurve.out", filt = 'SDSS_r')

                #write sample, avg_mag_diff, mag_mag_diff, peak_lum, peak_time, rise_time, fall_time to file
                write_to_file(lc_properties_file, f"{sid}\t{avg_mag_diff}\t{max_mag_diff}\t{peak_lum}\t{peak_time}\t{rise_time}\t{fall_time}\n", append = True)


        # Overall title for the whole figure
        fig.suptitle(f"Sample ID: {sid}", fontsize=24)

        # Tight layout so subplots don't overlap
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        mkdir(save_dir+f"{sid}/")
        fig.savefig(save_dir+f"{sid}/multi_band_lc.pdf", bbox_inches='tight')
        plt.close(fig)



        # Create one big figure with a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # adjust size as needed
        axes = np.array(axes).reshape(2, 2)            # ensure 2x2 indexing

        for idx, filt in enumerate(filt_list):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            if 'supp' in sid:
                sed_lc, sed_times = get_lc_from_file(dir1+""+str(sid)[5:]+"/lightcurve.out", filt = "SDSS_"+filt)
            else:
                sed_lc, sed_times = get_lc_from_file(dir1+""+str(sid)+"/lightcurve.out", filt = "SDSS_"+filt)
            if sed_lc is None:
                print("Light curve not found for sample id: "+str(sid))
                continue
            
            lc, times = get_lc_from_file(out_file, filt = "SDSS_"+filt)
            if lc is None:
                print("Light curve not found for sample id: "+str(sid))
                continue

            idx = np.where((times > 7) & (times < 40))
            sed_idx = np.where((sed_times > 7) & (sed_times < 40))
            avg_mag_diff = str(round((lc[idx] - sed_lc[sed_idx]).mean(), 2))
            max_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).max(), 2))
            mag_diff = lc[idx] - sed_lc[sed_idx]

            ax.plot(
                sed_times[sed_idx], mag_diff,
                label=f"Mag Diff {filt} LC",
                color=sKy_colors['blue'],
                alpha=0.7,
                linewidth=4,
            )
            
            if row == 0:
                ax.set_xlabel("")
                ax.set_xticks([])

            else:
                ax.set_xlabel("Time (Days)", fontsize=18)
                
            ax.set_ylabel("", fontsize=18)

            ax.set_title(f"{filt}-band, Avg|Max Mag Diff: "+str(avg_mag_diff)+" | "+str(max_mag_diff), fontsize=25)
            ax.legend(fontsize=12)

        # Overall title for the whole figure
        fig.suptitle(f"Sample ID: {sid}", fontsize=24)

        # Tight layout so subplots don't overlap
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])


        
        fig.savefig(save_dir+f"{sid}/mag_diff.pdf", bbox_inches='tight')
        plt.close(fig)



    lc_properties_file = save_dir+"/lc_properties.txt"
    lc_prop_dict = read_file(lc_properties_file)['dict']

    write_to_file(lc_properties_file, f"{sid}\t{avg_mag_diff}\t{max_mag_diff}\t{peak_lum}\t{peak_time}\t{rise_time}\t{fall_time}\n", append = True)

    sids = np.asarray(lc_prop_dict[0])
    rise_times = np.asarray(lc_prop_dict[5], dtype = float)
    fall_times = np.asarray(lc_prop_dict[6], dtype = float)
    avg_mag_diff = np.asarray(lc_prop_dict[1], dtype = float)
    max_mag_diff = np.asarray(lc_prop_dict[2], dtype = float)

    mkdir(save_dir+"/figs/")


    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize = 35)
    plt.ylabel("Average Mag Diff", fontsize = 35)
    for i in range(len(fall_times)):
        if np.abs(avg_mag_diff[i]) > 0.25:
            plt.text(fall_times[i], avg_mag_diff[i], f"{sids[i]}", fontsize=12, ha='center', va='bottom', weight='bold',bbox=bbox_props)
    plt.scatter(fall_times, avg_mag_diff, s = 300, color = sKy_colors['blue'])
    plt.savefig(save_dir+"/figs/avg_ft.pdf", bbox_inches = 'tight')
    plt.close()


    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize = 35)
    plt.ylabel("Maximum Mag Diff", fontsize = 35)
    for i in range(len(fall_times)):
        if np.abs(max_mag_diff[i]) > 0.25:
            plt.text(fall_times[i], max_mag_diff[i], f"{sids[i]}", fontsize=12, ha='center', va='bottom', weight='bold',zorder = 100,bbox=bbox_props)
    plt.scatter(fall_times, max_mag_diff, s = 300, color = sKy_colors['blue'])
    plt.savefig(save_dir+"/figs/max_ft.pdf", bbox_inches = 'tight')
    plt.close()


    
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize = 35)
    plt.ylabel("Average Mag Diff", fontsize = 35)
    for i in range(len(rise_times)):
        if np.abs(avg_mag_diff[i]) > 0.25 or rise_times[i] < 10:
            plt.text(rise_times[i], avg_mag_diff[i], f"{sids[i]}", fontsize=12, ha='center', va='bottom', weight='bold',zorder = 100,bbox=bbox_props)
    plt.scatter(rise_times, avg_mag_diff, s = 300, color = sKy_colors['blue'])
    plt.savefig(save_dir+"/figs/avg_rt.pdf", bbox_inches = 'tight')
    plt.close()


    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize = 35)
    plt.ylabel("Maximum Mag Diff", fontsize = 35)
    for i in range(len(rise_times)):
        if np.abs(max_mag_diff[i]) > 0.25:
            plt.text(rise_times[i], max_mag_diff[i], f"{sids[i]}", fontsize=12, ha='center', va='bottom', weight='bold',zorder = 100,bbox=bbox_props)
    plt.scatter(rise_times, max_mag_diff, s = 300, color = sKy_colors['blue'])
    plt.savefig(save_dir+"/figs/max_rt.pdf", bbox_inches = 'tight')
    plt.close()

    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize = 35)
    plt.hist(fall_times, color = sKy_colors['blue'], density = True)
    plt.savefig(save_dir+"/figs/ft_hist.pdf", bbox_inches = 'tight')
    plt.close()

    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize = 35)
    plt.hist(rise_times, color = sKy_colors['blue'], density = True)
    plt.savefig(save_dir+"/figs/rt_hist.pdf", bbox_inches = 'tight')
    plt.close()

