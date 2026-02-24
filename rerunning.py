import numpy as np
import os
import sys
from helpers import *
import glob
import subprocess
import shutil
from sedoNNa.utils import *

num_to_start = 30
job_prefix = 'supp_'

samples_to_run = []
started = 0


def decode_jobs(command, prefix = ''):
    result = subprocess.check_output(command).decode('utf-8')
    jobs = str(result).replace(" ", '').replace('\t', '').split('\n')
    samples = []
    for i in jobs:
        if len(prefix) > 0 and prefix not in i:
            continue
            
        if i is not None:
            samples.append(str(i))
    
    return samples

def get_jobs_running(prefix = ''):
    return decode_jobs(['squeue', '-u', 'kyadavalli', '-h', '-t', 'running', '-r', '-O', 'name'], prefix = prefix)

def get_jobs_pending(prefix = ''):
    return decode_jobs(['squeue', '-u', 'kyadavalli', '-h', '-t', 'pending', '-r', '-O', 'name'], prefix = prefix)


samples_running = get_jobs_running(prefix = job_prefix)
print("samples running: "+str(len(samples_running))+"\n"+str(samples_running))
samples_pending = get_jobs_pending(prefix = job_prefix)
print("samples pending: "+str(len(samples_pending))+"\n"+str(samples_pending))



dir1 = "supplemental_grid/"
sample_dirs = glob.glob(dir1+"/*/mod.mod")
print("found "+str(len(sample_dirs))+" samples")

for sample_dir in sample_dirs:
    samp = sample_dir.split('/')[-2]
    outdir = dir1+"/"+str(samp)+"/"
    if len(samples_to_run) > 0 and str(samp) not in samples_to_run:
        continue

    #copy a cleaning_script
    if not os.path.exists(outdir+"/cleaning_spec.py"):
        write_to_file(outdir+"/cleaning_spec.py", cleaning_spec_script, append = True)

    write_to_file(outdir+"/cleaning_up.py", cleaning_up_script, append = False)
    
    #check if this sample is running or pending
    if job_prefix+samp in samples_running or job_prefix+samp in samples_pending:
        
        continue
        
    #check if this sample is done
    if os.path.exists(outdir+"spectrum_final.dat"):
        continue
        
    if started >= num_to_start:
        continue
    
    print("copying batch file to sample "+str(samp))
    #copy a run_batch file
    batch_script = get_batch_script(job_name = job_prefix+samp, hours = 3, n = 64)
    write_to_file(outdir+"run_batch.sub", batch_script)

    
    #decide whether this sample needs to be restarted or started from scratch
    #find all the chk files in this folder, copy the last one to chk.h5
    chk_files = glob.glob(outdir+"/chk*")
    
    if len(chk_files) == 0:
        
        #this one needs to be started from scratch
        #copy the param file

        param_script = get_param_script(start_time = 5.0, stop_time = 50, dt = 0.2, restart = False)
        write_to_file(outdir+"param_lc_lte_exp.lua", param_script)

        
        
    else:
        last_updated = [os.path.getmtime(i) for i in chk_files]
        idx = find_nearest(last_updated, np.max(last_updated))
        chk_files_short_names = [os.path.basename(i) for i in chk_files]
        
        last_chk_file = chk_files_short_names[idx]
        if last_chk_file != 'chk.h5':
            try:
                shutil.move(chk_files[idx], outdir+"/chk.h5")
            except:
                print("Unable to move chk file in "+str(sample_to_restart)+", skipping this sample")
                continue
        all_chk_files = glob.glob(outdir+"/chk_*")
        if len(all_chk_files) > 0:
            for f in all_chk_files:
                rm(f)
        
        
        
        param_script = get_param_script(start_time = 5.0, stop_time = 50, dt = 0.2, restart = True)
        write_to_file(outdir+"param_lc_lte_exp.lua", param_script)
        
    os.system("cd "+outdir+"; mv log.log old_log.log; sbatch run_batch.sub; sleep .5")
    print("Starting "+str(samp))
    started += 1
    
    
    
    
    
    
