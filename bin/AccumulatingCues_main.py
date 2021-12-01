"""
##
Code for the Evidence Accumulation task in Figure 3C (Liu et al., PNAS, 2021) 
Code adapted by Yuhan Helena Liu, PhD Candidate, University of Washingto
Code adapted from https://github.com/IGITUGraz/LSNN-official
    with the following copyright message retained from the original code:

##
The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik, einsum_bi_bijk_to_bjk

import os
import random 
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.toolbox.file_saver_dumper_no_h5py import save_file, load_file, get_storage_path_reference

import json
from rewiring_tools_NP import rewiring_optimizer_wrapper_NP, rewiring_optimizer_wrapper, weight_sampler
from spiking_models_NP import tf_cell_to_savable_dict, exp_convolve, ALIF, LIF 

#%matplotlib inline

FLAGS = tf.app.flags.FLAGS
print(tf.__version__)

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_bool('save_data', False, 'whether to save simulation data in result folder')
##
tf.app.flags.DEFINE_integer('n_batch_train',  64, 'size of the training minibatch')
tf.app.flags.DEFINE_integer('n_batch_validation', 64, 'size of the validation minibatch')
tf.app.flags.DEFINE_integer('n_in', 40, 'number of input units')
tf.app.flags.DEFINE_integer('n_regular', 50, 'number of regular spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('n_adaptive', 50, 'number of adaptive spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('n_iter', 2000, 'number of training iterations') # 500 for Fig 3
tf.app.flags.DEFINE_integer('n_delay', 1, 'maximum synaptic delay')
tf.app.flags.DEFINE_integer('n_ref', 5, 'number of refractory steps')
tf.app.flags.DEFINE_integer('lr_decay_every', 200000, 'decay learning rate every lr_decay_every steps')
tf.app.flags.DEFINE_integer('print_every', 10, 'frequency of validation')
tf.app.flags.DEFINE_bool('NP', True, 'Use Neuropeptide or BPTT')
tf.app.flags.DEFINE_integer('NP_mode', 3, 'when NP==True: 1-eprop, 2-MDGLbca, 3-MDGL, 5-NLMDGL, 7-OnMDGL')
##
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold')
# to solve a task successfully we usually set tau_a to be close to the expected delay / memory length needed
tf.app.flags.DEFINE_float('tau_a', 2000, 'Adaptation time constant')
tf.app.flags.DEFINE_integer('delay_window', 850, 'Delay Window')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of output readouts')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('learning_rate', 0.0025, 'Base learning rate')
##
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.8, 'proportion of excitatory neurons')
##
tf.app.flags.DEFINE_bool('verbose', True, 'Print many info during training')
tf.app.flags.DEFINE_bool('neuron_sign', True,
                         "If rewiring is active, this will fix the sign of neurons (Dale's law)")

tf.app.flags.DEFINE_float('rewiring_connectivity', 0.1, 'max connectivity limit in the network (-1 turns off DEEP R)')
tf.app.flags.DEFINE_float('wout_connectivity', 0.1, 'similar to above but for output weights')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization used in rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')
# Analog values are fed to only single neuron

# Define the flag object as dictionnary for saving purposes
_, storage_path, flag_dict = get_storage_path_reference(__file__, FLAGS, './results/', flags=False,
                                                        comment=len(FLAGS.comment) > 0)
if FLAGS.save_data:
    os.makedirs(storage_path, exist_ok=True)
    save_file(flag_dict, storage_path, 'flag', 'pickle')
    print('saving data to: ' + storage_path)
print(json.dumps(flag_dict, indent=4))

dt = 1.  # Time step is by default 1 ms
n_output_symbols = 2 # left or right 
n_cue = 7       # number of cues received along the track (up to 7)
delay_window = FLAGS.delay_window
cue_window = 150
T = n_cue*150+(delay_window-50)+cue_window
n_unit = FLAGS.n_regular+FLAGS.n_adaptive

eprop = (FLAGS.NP_mode == 1)
Trunc = (FLAGS.NP_mode == 2)
EI_approx = (FLAGS.NP_mode == 3) 
all_j = (FLAGS.NP_mode == 5)
EI_causal = (FLAGS.NP_mode == 7)

assert FLAGS.n_in==40, 'the present model assumes 40 input units'
assert FLAGS.n_delay==1, 'the present model assumes synaptic delay=1'

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(FLAGS.proportion_excitatory * FLAGS.n_in) + 1
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(FLAGS.proportion_excitatory * (FLAGS.n_regular + FLAGS.n_adaptive)) + 1
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not (FLAGS.neuron_sign == False): print(
        'WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Define the network
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=FLAGS.tau_v, n_delay=FLAGS.n_delay,
            n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=FLAGS.tau_a, beta=beta, thr=FLAGS.thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity, 
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor)

# Generate input
inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes') # MAIN input spike placeholder

targets = tf.placeholder(dtype=tf.int64, shape=(None,),
                         name='Targets')  # Lists of target characters of the recall task

n_unit = FLAGS.n_regular + FLAGS.n_adaptive

def get_data_dict(batch_size, n_cue, tr_len):
    """
    Generate the dictionary to be fed when running a tensorflow op.
    Assumes n_in=40, each population with 10 units 
    """
    assert(n_cue<=7) # number of cues received along the track up to 7 
    assert(batch_size<128)    
    # set time windows
    n_perm = 2**n_cue     # possible permutations

    # Initialize target and input cue matrices 
    target_num = -1*np.ones((batch_size,))
    
    cue_batch = np.random.permutation(128) % n_perm
    cue_batch = cue_batch[0:batch_size]
   
    spike_stack=np.zeros((batch_size,tr_len,40))
    
    # Get spike encoding and target for each trial     
    def get_spike_stack(cue_list):  # spikes per example
        i4_spike = np.random.poisson(0.01, (T,10))  # population 4, 10Hz throughout the 2000ms window
        i3_spike = np.concatenate((np.zeros((tr_len-150,10)), \
                                   np.random.poisson(0.04,(150,10))),0)
            
        i1_spike = np.zeros((tr_len,10))  
        i2_spike = np.zeros((tr_len,10))
        
        # loop through cues within a trial 
        for cc in range(len(cue_list)):
            tstamps = np.array(range(cc*150,cc*150+100))  # cue window
            if cue_list[cc] == 1:    # left
                i1_spike[tstamps,:] = np.random.poisson(0.04, (100,10))
            else:       # right
                i2_spike[tstamps,:] = np.random.poisson(0.04, (100,10))
            
        spike_stack = np.concatenate((i1_spike,i2_spike,i3_spike,i4_spike),1)
        target_dir=int(np.sum(cue_list)>(len(cue_list)/2))   # 1 for left
        
        return spike_stack, target_dir

    # loop through trials across batches
    for tr in range(len(cue_batch)):
        cue_num = cue_batch[tr] 
        if cue_num >= (n_perm/2):
            cue_list = np.array([int(i) for i in bin(cue_num)[2:]])
        else:
            cue_list = np.array([int(i) for i in bin(cue_num)[2:]])
            cue_list = np.concatenate((np.zeros((n_cue-len(cue_list),)), cue_list))
            
        spike_stack[tr,:,:], target_num[tr] = get_spike_stack(cue_list)
            
    # transform target one hot from batch x classes to batch x time x classes
    data_dict = {inputs: spike_stack, targets: target_num}
    return data_dict, cue_batch

outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
if FLAGS.NP: 
    z, v, b, psi, bpsi, e_trace, e_trace_i, i_syn = outputs
else: 
    z, v, b, psi, _, _, _, _ = outputs


z_regular = z[:, :, :FLAGS.n_regular]
z_adaptive = z[:, :, FLAGS.n_regular:]
av = tf.reduce_mean(z, axis=(0, 1)) / dt

with tf.name_scope('ClassificationLoss'):
    psp_decay = np.exp(-dt / FLAGS.tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
    psp = exp_convolve(z, decay=psp_decay)
    n_neurons = z.get_shape()[2]

    # Define the readout weights           
    if (0 < FLAGS.wout_connectivity):
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                     FLAGS.wout_connectivity,
                                                     neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.Variable(np.random.randn(n_unit, n_output_symbols) / np.sqrt(n_unit),
                        name='out_weight', dtype=tf.float32)
    b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer())

    # Define the loss function   
    out = einsum_bij_jk_to_bik(psp, w_out) + b_out # (n_batch, n_time, n_class)    
    
    Y_predict = out[:, -1, :]   
    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=Y_predict))
    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)

    # Define the accuracy
    Y_predict_num = tf.argmax(Y_predict, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))
    
if FLAGS.NP:
    # Setup
    batch_size = FLAGS.n_batch_train
    rho=np.exp(-dt/FLAGS.tau_a)
    alp=cell._decay[0]     # TODO: fix this later 
    kap=alp
 
    e_trace = tf.reshape(e_trace, [tf.shape(e_trace)[0],tf.shape(e_trace)[1],n_unit,n_unit])  #(b,t,p,q)  
    e_trace_i = tf.reshape(e_trace_i, \
                                [tf.shape(e_trace_i)[0],tf.shape(e_trace_i)[1],n_unit,FLAGS.n_in])         
    partial_Ez = tf.gradients(loss_recall, z)[0] 
 
    if eprop: 
        partial_Ez_ = tf.expand_dims(partial_Ez, axis=-1) 
        dEdWr = tf.reduce_sum(tf.multiply(partial_Ez_, e_trace), axis=(0,1)) 
        dEdWi = tf.reduce_sum(tf.multiply(partial_Ez_, e_trace_i), axis=(0,1)) 
        
    else:                
        if EI_approx:                       
            # weight average 
            # A bit confusing, W stored as (p,j)
            Wrec = cell.W_rec[:,:,0] #(p,j)  
            Wii = tf.reduce_sum(Wrec[:n_inhibitory,:n_inhibitory])/tf.count_nonzero(Wrec[:n_inhibitory,:n_inhibitory],dtype=tf.float32) 
            Wei = tf.reduce_sum(Wrec[n_inhibitory:,:n_inhibitory])/tf.count_nonzero(Wrec[n_inhibitory:,:n_inhibitory],dtype=tf.float32) 
            Wie = tf.reduce_sum(Wrec[:n_inhibitory,n_inhibitory:])/tf.count_nonzero(Wrec[:n_inhibitory,n_inhibitory:],dtype=tf.float32) 
            Wee = tf.reduce_sum(Wrec[n_inhibitory:,n_inhibitory:])/tf.count_nonzero(Wrec[n_inhibitory:,n_inhibitory:],dtype=tf.float32) 
            
            a_tj = tf.multiply(partial_Ez[:,1:], psi[:,1:])    
            # Sum a over j \in alpha, & j connected to p
            Wadj = tf.where(tf.not_equal(Wrec,0), tf.ones_like(Wrec), tf.zeros_like(Wrec))
            sum_aE = einsum_bij_jk_to_bik(a_tj[:,:,n_inhibitory:],tf.transpose(Wadj[:,n_inhibitory:])) #btj,jp->btp, j in E
            sum_aI = einsum_bij_jk_to_bik(a_tj[:,:,:n_inhibitory],tf.transpose(Wadj[:,:n_inhibitory])) #j in I
            sum_aE = tf.expand_dims(sum_aE, axis=-1) #(b,t,p,1)
            sum_aI = tf.expand_dims(sum_aI, axis=-1)
            
            # outer sum, sum over alpha 
            gamEq = e_trace[:,:-1,n_inhibitory:] * (Wee*sum_aE[:,:,n_inhibitory:] + Wei*sum_aI[:,:,n_inhibitory:]) #(b,t,p,q)*(b,t,p,1), p in E
            gamIq = e_trace[:,:-1,:n_inhibitory] * (Wie*sum_aE[:,:,:n_inhibitory] + Wii*sum_aI[:,:,:n_inhibitory]) # p in I
            gam = tf.reduce_sum(tf.concat([gamIq, gamEq], axis=2), axis=(0,1))
            gamEq_i = e_trace_i[:,:-1,n_inhibitory:] * (Wee*sum_aE[:,:,n_inhibitory:] + Wei*sum_aI[:,:,n_inhibitory:]) 
            gamIq_i = e_trace_i[:,:-1,:n_inhibitory] * (Wie*sum_aE[:,:,:n_inhibitory] + Wii*sum_aI[:,:,:n_inhibitory])
            gam_i = tf.reduce_sum(tf.concat([gamIq_i, gamEq_i], axis=2), axis=(0,1))
        
        elif all_j:
            Wrec = cell.W_rec[:,:,0] #(p,j)
            Wii = tf.reduce_mean(Wrec[:n_inhibitory,:n_inhibitory])
            Wei = tf.reduce_mean(Wrec[n_inhibitory:,:n_inhibitory])
            Wie = tf.reduce_mean(Wrec[:n_inhibitory,n_inhibitory:])
            Wee = tf.reduce_mean(Wrec[n_inhibitory:,n_inhibitory:])
            
            a_tj = tf.multiply(partial_Ez[:,1:], psi[:,1:])
            sum_aE = tf.expand_dims(tf.expand_dims(tf.reduce_sum(a_tj[:,:,n_inhibitory:], axis=-1), axis=-1),axis=-1) #(b,t,1,1), j in E
            sum_aI = tf.expand_dims(tf.expand_dims(tf.reduce_sum(a_tj[:,:,:n_inhibitory], axis=-1), axis=-1),axis=-1) #j in I
            
            gamEq = e_trace[:,:-1,n_inhibitory:] * (Wee*sum_aE + Wei*sum_aI) # (b,t,p,q)*(b,t,1,1), p in E
            gamIq = e_trace[:,:-1,:n_inhibitory] * (Wie*sum_aE + Wii*sum_aI) # p in I
            gam = tf.reduce_sum(tf.concat([gamIq, gamEq], axis=2), axis=(0,1))
            gamEq_i = e_trace_i[:,:-1,n_inhibitory:] * (Wee*sum_aE + Wei*sum_aI) 
            gamIq_i = e_trace_i[:,:-1,:n_inhibitory] * (Wie*sum_aE + Wii*sum_aI)
            gam_i = tf.reduce_sum(tf.concat([gamIq_i, gamEq_i], axis=2), axis=(0,1))
        
        elif Trunc:  
            Wjp = tf.transpose(cell.W_rec[:,:,0]) #(p,j) -> (j,p)
            a_tj = tf.multiply(partial_Ez[:,1:], psi[:,1:])           
            sum_a = tf.expand_dims(einsum_bij_jk_to_bik(a_tj, Wjp), axis=-1) #(b,t,p,1)
            
            # outer sum, sum over alpha 
            gam = e_trace[:,:-1] * sum_a
            gam_i = e_trace_i[:,:-1] * sum_a
            gam_8batch = (1-alp)*tf.reduce_sum(gam[0:8], axis=(0,1)) 
            gam_i_8batch = (1-alp)*tf.reduce_sum(gam_i[0:8], axis=(0,1)) 
            gam = tf.reduce_sum(gam, axis=(0,1)) 
            gam_i = tf.reduce_sum(gam_i, axis=(0,1)) 
            
        elif EI_causal:
            partial_Efiltz = tf.gradients(loss_recall, psp)[0]
            filtering = lambda filtered_g, g: filtered_g * psp_decay + g * (1-psp_decay) 
            #summing = lambda summed_g, g: summed_g + g  
            psi_tmajor = tf.transpose(psi, perm=[1, 0, 2]) #(t,b,j)
            psi_0 = tf.zeros_like(psi_tmajor[0])
            filtered_psi = tf.scan(filtering, psi_tmajor, initializer=psi_0)
            filtered_psi = tf.transpose(filtered_psi, perm=[1, 0, 2])
            a_tj = tf.multiply(partial_Efiltz[:,1:], filtered_psi[:,1:]) 
            
            e_tmajor = tf.transpose(e_trace, perm=[1, 0, 2, 3]) #(t,b,p,q)
            ei_tmajor = tf.transpose(e_trace_i, perm=[1, 0, 2, 3])
            e_0 = tf.zeros_like(e_tmajor[0])
            ei_0 = tf.zeros_like(ei_tmajor[0])
            filtered_e = tf.scan(filtering, e_tmajor, initializer=e_0)
            filtered_ei = tf.scan(filtering, ei_tmajor, initializer=ei_0)
            filtered_e = tf.transpose(filtered_e, perm=[1, 0, 2, 3])
            filtered_ei = tf.transpose(filtered_ei, perm=[1, 0, 2, 3])
            
            Wrec = cell.W_rec[:,:,0]
            Wadj = tf.where(tf.not_equal(Wrec,0), tf.ones_like(Wrec), tf.zeros_like(Wrec))
            Wii = tf.reduce_sum(Wrec[:n_inhibitory,:n_inhibitory])/tf.count_nonzero(Wrec[:n_inhibitory,:n_inhibitory],dtype=tf.float32) 
            Wei = tf.reduce_sum(Wrec[n_inhibitory:,:n_inhibitory])/tf.count_nonzero(Wrec[n_inhibitory:,:n_inhibitory],dtype=tf.float32) 
            Wie = tf.reduce_sum(Wrec[:n_inhibitory,n_inhibitory:])/tf.count_nonzero(Wrec[:n_inhibitory,n_inhibitory:],dtype=tf.float32) 
            Wee = tf.reduce_sum(Wrec[n_inhibitory:,n_inhibitory:])/tf.count_nonzero(Wrec[n_inhibitory:,n_inhibitory:],dtype=tf.float32) 
                
            sum_aE = einsum_bij_jk_to_bik(a_tj[:,:,n_inhibitory:],tf.transpose(Wadj[:,n_inhibitory:]))
            sum_aI = einsum_bij_jk_to_bik(a_tj[:,:,:n_inhibitory],tf.transpose(Wadj[:,:n_inhibitory]))
            sum_aE = tf.expand_dims(sum_aE, axis=-1)
            sum_aI = tf.expand_dims(sum_aI, axis=-1)
            
            # outer sum, sum over alpha 
            gamEq = filtered_e[:,:-1,n_inhibitory:] * (Wee*sum_aE[:,:,n_inhibitory:] + Wei*sum_aI[:,:,n_inhibitory:]) 
            gamIq = filtered_e[:,:-1,:n_inhibitory] * (Wie*sum_aE[:,:,:n_inhibitory] + Wii*sum_aI[:,:,:n_inhibitory]) 
            gam = tf.reduce_sum(tf.concat([gamIq, gamEq], axis=2), axis=(0,1))
            gamEq_i = filtered_ei[:,:-1,n_inhibitory:] * (Wee*sum_aE[:,:,n_inhibitory:] + Wei*sum_aI[:,:,n_inhibitory:]) 
            gamIq_i = filtered_ei[:,:-1,:n_inhibitory] * (Wie*sum_aE[:,:,:n_inhibitory] + Wii*sum_aI[:,:,:n_inhibitory])
            gam_i = tf.reduce_sum(tf.concat([gamIq_i, gamEq_i], axis=2), axis=(0,1))
        
        partial_Ez_ = tf.expand_dims(partial_Ez, axis=-1) 
        dEdWr = tf.reduce_sum(tf.multiply(partial_Ez_, e_trace), axis=(0,1)) + (1-alp)*gam
        dEdWi = tf.reduce_sum(tf.multiply(partial_Ez_, e_trace_i), axis=(0,1)) + (1-alp)*gam_i 
    
    dEdWr = tf.transpose(dEdWr) # (p,q) -> (q,p); i.e. (post,pre) -> (pre,post)
    dEdWi = tf.transpose(dEdWi)       
    
   
# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)  # Op to decay learning rate

    loss = loss_recall

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:
        if FLAGS.NP:
            train_step = rewiring_optimizer_wrapper_NP(dEdWi, dEdWr, \
                            optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                    FLAGS.rewiring_connectivity,
                                                    global_step=global_step,
                                                    var_list=tf.trainable_variables())            
        else: 
            train_step = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                    FLAGS.rewiring_connectivity,
                                                    global_step=global_step,
                                                    var_list=tf.trainable_variables())
    else:
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

# Real-time plotting
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Store some results across iterations
test_loss_list = []
test_loss_tot_list = []
test_error_list = []
training_time_list = []
time_to_ref_list = []
iter_list = []

# Dictionaries of tensorflow ops to be evaluated simultaneously by a session
results_tensors = {'loss': loss,
                   'loss_recall': loss_recall,
                   'accuracy': accuracy,
                   'av': av,
                   'learning_rate': learning_rate,   
                   'w_in_val': cell.w_in_val,
                   'w_rec_val': cell.w_rec_val,
                   'w_out': w_out,
                   'b_out': b_out
                   }

plot_result_tensors = {'input_spikes': inputs,
                       'z': z,
                       'v': v,
                       'psp': psp,
                       'out_plot': out_plot,
                       'Y_predict': Y_predict,
                       'z_regular': z_regular,
                       'z_adaptive': z_adaptive,
                       'targets': targets,
                       'b_out': b_out}

t_train = 0
for k_iter in range(FLAGS.n_iter+1):

    # Decaying learning rate
    if k_iter > 1 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))          
    
    # Print some values to monitor convergence
    if np.mod(k_iter, FLAGS.print_every) == 0:

        val_dict, input_cue = get_data_dict(FLAGS.n_batch_validation, n_cue, T)
        results_values, plot_results_values = \
            sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)

        # Storage of the results
        test_loss_list.append(results_values['loss_recall'])
        test_loss_tot_list.append(results_values['loss'])
        test_error_list.append(results_values['accuracy'])
        training_time_list.append(t_train)
        iter_list.append(k_iter)

        print(
            '''Iteration {}, validation accuracy {:.3g} '''
                .format(k_iter, test_error_list[-1],))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(results_values['av'] * 1000)

        # some connectivity statistics
        rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)
        empirical_connectivities = [nz / size for nz, size in zip(non_zeros, sizes)]

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            connectivity (total {:.3g})\t W_in {:.3g} \t W_rec {:.2g} \t\t w_out {:.2g}
            number of non zero weights \t W_in {}/{} \t W_rec {}/{} \t w_out {}/{}

            Classification {:.2g} 
            learning rate {:.2g} \t training op. time {:.2g}s
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                empirical_connectivity,
                empirical_connectivities[0], empirical_connectivities[1], empirical_connectivities[2],
                non_zeros[0], sizes[0],
                non_zeros[1], sizes[1],
                non_zeros[2], sizes[2],
                results_values['loss_recall'], 
                results_values['learning_rate'], t_train,
            ))

        # Save files result
        if FLAGS.save_data:
            results = {
                'error_list': test_error_list,
                'loss_list': test_loss_list, 
                'loss_tot_list': test_loss_tot_list,
                'iter_list': iter_list, 
                'av': results_values['av'],
                'flags': flag_dict,
            }            
                
            save_file(results, storage_path, 'results', file_type='pickle')     

    # Train
    t0 = time()
    train_dict, input_cue = get_data_dict(FLAGS.n_batch_train, n_cue, T)
    
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0    
    # print('lala')


# Print some stuff
print('Test accuracies over validation trials:')
print(test_error_list)
print('Test rescall loss:')
print(test_loss_list)

# Plot loss and accuracy over iter
#plot_loss_acc(iter_list,test_loss_list,test_error_list,plt)
