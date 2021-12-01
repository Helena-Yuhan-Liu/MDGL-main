# -*- coding: utf-8 -*-
"""
Code for plotting the learning curves of saved runs

@author: Yuhan Helena Liu, PhD Candidate, University of Washington
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from file_saver_dumper_no_h5py import save_file, load_file, get_storage_path_reference
import json
import os

## Setup
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16 
M = 5 # moving avg parametner, odd number   

# Paths 
results_path = './results/AccumulatingCues_main/'
file_name = 'results'     
    
file_name = 'results'
sim_list=['default']
sim_mode = sim_list[0]   


## Plot results
def movingmean(data_set, periods=3):
    data_set = np.array(data_set)
    if periods > 1:
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid') 
    else:
        return data_set
    
def iter_loss_acc(results_path, file_name, M, comment):
    all_f = os.listdir(results_path)
    flist = []
    for f in range(len(all_f)):
        if comment in all_f[f]:
            flist.append(all_f[f])
    
    plot_len = -1
    av_list =[]
    for f in range(len(flist)):
        file_path = results_path + flist[f]
        results_ = load_file(file_path,file_name,file_type='pickle')
        if f==0:
            iterlist = results_['iter_list'][0:plot_len]     
            loss = np.expand_dims(movingmean(results_['loss_tot_list'][0:plot_len] ,M),axis=0) 
        else:
            if 'default_BPTT' in comment:
                trial_loss = np.expand_dims(movingmean(results_['loss_tot_list'][::2][0:plot_len] ,M),axis=0)
            else: 
                trial_loss = np.expand_dims(movingmean(results_['loss_tot_list'][0:plot_len] ,M),axis=0)
            loss = np.concatenate((loss, trial_loss), axis=0) 
    
    mean_loss = np.mean(loss, axis=0)
    std_loss = np.std(loss, axis=0,ddof=1)
    iterlist = iterlist[int(M/2):-int(M/2)]
    return iterlist, mean_loss, std_loss #, np.array(av_list)


comment_list = [[sim_mode+'_Eprop', 'm', (0.9, 0.75, 0.9), 'e-prop'],            # The three algorithms
                [sim_mode+'_BPTT', 'k', (0.75, 0.75, 0.75),'BPTT'] ,
                [sim_mode+'_EI', 'g', (0.75, 0.9, 0.75),'MDGL'] ]


fig0 = plt.figure()
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 
for ii in range(len(comment_list)):
    comm_iterlist, mean_comm, std_comm = iter_loss_acc(results_path, file_name, M, comment_list[ii][0])
    plt.plot(comm_iterlist, mean_comm, color=comment_list[ii][1],label=comment_list[ii][3]) #'E-prop
    plt.fill_between(comm_iterlist, mean_comm-std_comm, mean_comm+std_comm,color=comment_list[ii][2])
    
plt.legend();
plt.xlabel('Training Iterations')
plt.ylabel('Loss');
plt.title('Evidence Accumulation')
fig0.savefig(os.path.join(results_path, 'LearningCurve_' + sim_mode + '_AccuCues.pdf'), format='pdf')
