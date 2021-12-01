"""
##
Code modified by Yuhan Helena Liu, PhD Candidate, University of Washington
Modified to compute and store the eligibility trace 
Modified from https://github.com/IGITUGraz/LSNN-official
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

from distutils.version import LooseVersion
import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor

if LooseVersion(tf.__version__) >= LooseVersion("1.11"):
    from tensorflow.python.ops.variables import Variable, RefVariable
else:
    print("Using tensorflow version older then 1.11 -> skipping RefVariable storing")
    from tensorflow.python.ops.variables import Variable

from rewiring_tools_NP import weight_sampler
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bi_ijk_to_bjk, einsum_bij_jk_to_bik
from lsnn.toolbox.tensorflow_utils import tf_roll

from time import time

Cell = tf.contrib.rnn.BasicRNNCell


def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            if np.shape(v) == ():
                v = [v]
            shape = np.concatenate([[batch_size], v])
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)

        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v})
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update


#################################
# Spike function
#################################

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor 

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.equal(d, kd)
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer
def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.

    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    tftypes = [Variable, Tensor]
    if LooseVersion(tf.__version__) >= LooseVersion("1.11"):
        tftypes.append(RefVariable)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in tftypes:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.int64, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute of key {} and value {} has type {}, recoding it as string.'.format(k, v, type(v)))
            dict_to_save[k] = str(v)

    return dict_to_save


class LIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1, rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 dampening_factor=0.3,
                 injected_noise_current=0.,
                 V0=1.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.

        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
        """

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor

        # Parameters
        self.n_delay = n_delay
        self.n_refractory = n_refractory

        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tf.Variable(tau, dtype=dtype, name="Tau", trainable=False)
        self._decay = tf.exp(-dt / tau)
        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)

        self.V0 = V0
        self.injected_noise_current = injected_noise_current

        self.rewiring_connectivity = rewiring_connectivity
        self.in_neuron_sign = in_neuron_sign
        self.rec_neuron_sign = rec_neuron_sign

        with tf.variable_scope('InputWeights'):

            # Input weights
            if 0 < rewiring_connectivity < 1:
                self.w_in_val, self.w_in_sign, self.w_in_var, self.w_in_inicon = weight_sampler(n_in, n_rec, rewiring_connectivity,
                                                                                 neuron_sign=in_neuron_sign)
            else:
                self.w_in_var = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputWeight")
                self.w_in_val = self.w_in_var

            self.w_in_val = self.V0 * self.w_in_val
            self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),
                                          dtype=tf.int64, name="InDelays", trainable=False)
            self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

        with tf.variable_scope('RecWeights'):
            if 0 < rewiring_connectivity < 1:
                self.w_rec_val, self.w_rec_sign, self.w_rec_var, self.w_rec_inicon = weight_sampler(n_rec, n_rec,
                                                                                    rewiring_connectivity,
                                                                                    neuron_sign=rec_neuron_sign)
            else:
                if rec_neuron_sign is not None or in_neuron_sign is not None:
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.w_rec_var = Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype,
                                          name='RecurrentWeight')
                self.w_rec_val = self.w_rec_var

            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

            self.w_rec_val = self.w_rec_val * self.V0
            self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
                                      self.w_rec_val)  # Disconnect autotapse
            self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec),
                                           dtype=tf.int64, name="RecDelays", trainable=False)
            self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    @property
    def state_size(self):
        return LIFStateTuple(v=self.n_rec,
                             z=self.n_rec,
                             i_future_buffer=(self.n_rec, self.n_delay),
                             z_buffer=(self.n_rec, self.n_refractory))

    @property
    def output_size(self):
        return self.n_rec

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return LIFStateTuple(
            v=v0,
            z=z0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
                state.z, self.W_rec)

        new_v, new_z = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = LIFStateTuple(v=new_v,
                                  z=new_z,
                                  i_future_buffer=new_i_future_buffer,
                                  z_buffer=new_z_buffer)
        return new_z, new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, thr_prev = None, decay=None, n_refractory=None, add_current=0.):
        """
        Function that generate the next spike and voltage tensor for given cell state.
        :param v
        :param z
        :param z_buffer:
        :param i_future_buffer:
        :param thr:
        :param decay:
        :param n_refractory:
        :param add_current:
        :return:
        """

        if self.injected_noise_current > 0:
            add_current = tf.random_normal(shape=z.shape, stddev=self.injected_noise_current)

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if thr_prev is None: thr_prev = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            I_reset = z * self.thr * self.dt

            new_v = decay * v + (1 - decay) * i_t - I_reset

            # Spike generation
            v_scaled = (new_v - thr) / self.thr     

            # new_z = differentiable_spikes(v_scaled=v_scaled)
            new_z = SpikeFunction(v_scaled, self.dampening_factor)          

            if n_refractory > 0:
                is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), 0)
                new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)
                # psi = tf.where(is_ref, tf.zeros_like(psi), psi)

            new_z = new_z * 1 / self.dt            
            psi = tf.gradients(new_z, new_v)[0] 

            return new_v, new_z , psi


ALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',
    # 'etrace_bar', 'etrace_bar_i',    
    'evec_v', 'evec_a', 'evec_v_i', 'evec_a_i',
    'psi', 'bpsi',
    'i_future_buffer',
    'z_buffer'))


class ALIF(LIF):
    def __init__(self, n_in, n_rec, tau=20, thr=0.01,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
                 tau_adaptation=200., beta=1.6,
                 rewiring_connectivity=-1, dampening_factor=0.3,
                 in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
                 V0=1.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.

        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param tau_adaptation: adaptation time constant for the threshold voltage
        :param beta: amplitude of adpatation
        :param rewiring_connectivity: number of non-zero synapses in weight matrices (at initialization)
        :param in_neuron_sign: vector of +1, -1 to specify input neuron signs
        :param rec_neuron_sign: same of recurrent neurons
        :param injected_noise_current: amplitude of current noise
        :param V0: to choose voltage unit, specify the value of V0=1 Volt in the desired unit (example V0=1000 to set voltage in millivolts)
        """

        super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
                                   dtype=dtype, n_delay=n_delay,
                                   rewiring_connectivity=rewiring_connectivity,
                                   dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign,
                                   rec_neuron_sign=rec_neuron_sign,
                                   injected_noise_current=injected_noise_current,
                                   V0=V0)

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.tau_adaptation = tf.Variable(tau_adaptation, dtype=dtype, name="TauAdaptation", trainable=False)

        self.beta = tf.Variable(beta, dtype=dtype, name="Beta", trainable=False)
        self.decay_b = np.exp(-dt / tau_adaptation)

    @property
    def output_size(self):  
        return [self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec, \
                self.n_rec*self.n_rec, self.n_rec*self.n_in,self.n_rec]

    @property
    def state_size(self):
        return ALIFStateTuple(v=self.n_rec,
                              z=self.n_rec,
                              b=self.n_rec,
                              evec_v=(self.n_rec, self.n_rec),
                              evec_a=(self.n_rec, self.n_rec),
                              evec_v_i=(self.n_rec, self.n_in),
                              evec_a_i=(self.n_rec, self.n_in),
                              psi=self.n_rec, 
                              bpsi=self.n_rec,
                              i_future_buffer=(self.n_rec, self.n_delay),
                              z_buffer=(self.n_rec, self.n_refractory))

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        
        evec0 = tf.zeros(shape=(batch_size, n_rec,n_rec), dtype=dtype)
        evec_i0 = tf.zeros(shape=(batch_size, n_rec, self.n_in), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return ALIFStateTuple(
            v=v0,
            z=z0,
            b=b0,            
            evec_v=evec0, 
            evec_a=evec0,
            evec_v_i=evec_i0, 
            evec_a_i=evec_i0,
            psi=v0,
            bpsi=v0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        with tf.name_scope('ALIFcall'):     
            i_syn = einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(state.z, self.W_rec)  
            i_future_buffer = state.i_future_buffer + i_syn
    
            new_b = self.decay_b * state.b + (1. - self.decay_b) * state.z
    
            thr = self.thr + new_b * self.beta # * self.V0            
            thr_prev = self.thr + state.b * self.beta 
    
            new_v, new_z, new_psi_ = self.LIF_dynamic(
                v=state.v,
                z=state.z,
                z_buffer=state.z_buffer,
                i_future_buffer=i_future_buffer,
                decay=self._decay,
                thr=thr, thr_prev=thr_prev)            
                
            # setup evec, etrace calculation 
            rho=self.decay_b #tf.exp(-self.dt/1500)
            alp=self._decay[0]      # Assume scalar
            kap=alp       
            beta_vec = tf.reshape(self.beta,[1,self.n_rec,1])
            psi_ = tf.expand_dims(state.psi, axis=-1)  #(b,j,1) # get evec from states at t-1 (13)
            bpsi_ = tf.expand_dims(state.bpsi, axis=-1)
            new_psi = tf.expand_dims(new_psi_, axis=-1)            
            new_bpsi_ = tf.gradients(new_z, new_b)[0]
            new_bpsi = tf.expand_dims(new_bpsi_,axis=-1)
            thr_ = tf.reshape(self.thr,[1,self.n_rec,1])
            
            # e_vec update 
            A11 = alp-tf.multiply(thr_, psi_)
            A12 = -tf.multiply(thr_, bpsi_) #*0
            A21 = (1-rho)*psi_ #*0
            A22 = (rho + (1-rho)*bpsi_) #*0      
            is_adapt = tf.broadcast_to(tf.greater(beta_vec,0),\
                                        [tf.shape(A12)[0],self.n_rec, 1])
            A12 = tf.where(is_adapt, A12, tf.zeros_like(A12))
            A21 = tf.where(is_adapt, A21, tf.zeros_like(A21))
            A22 = tf.where(is_adapt, A22, tf.zeros_like(A22))
                
            b_vec = (1-alp)*tf.expand_dims(state.z, axis=-2) #(b, 1, j)            
            new_evec_v = tf.add(tf.add(tf.multiply(A11,state.evec_v),\
                               tf.multiply(A12,state.evec_a)), b_vec)
            new_evec_a = tf.add(tf.multiply(A21,state.evec_v), tf.multiply(A22,state.evec_a))    
            

            # get e_trace
            partial_zh_v = new_psi
            partial_zh_a = new_bpsi
            etrace = tf.add(tf.multiply(partial_zh_v, new_evec_v), \
                          tf.multiply(partial_zh_a, new_evec_a)) 
            
            # repeat for Wi            
            b_vec_i = (1-alp)*tf.expand_dims(inputs, axis=-2) 
            new_evec_v_i = tf.add(tf.add(tf.multiply(A11,state.evec_v_i),\
                               tf.multiply(A12,state.evec_a_i)), b_vec_i)
            new_evec_a_i = tf.add(tf.multiply(A21,state.evec_v_i), tf.multiply(A22,state.evec_a_i))      
            etrace_i = tf.add(tf.multiply(partial_zh_v, new_evec_v_i), \
                          tf.multiply(partial_zh_a, new_evec_a_i))
            
            # reshape for the output
            e_trace_out = tf.reshape(etrace, \
                            [tf.shape(etrace)[0], self.n_rec*self.n_rec])
            e_trace_i_out = tf.reshape(etrace_i, \
                            [tf.shape(etrace_i)[0], self.n_rec*self.n_in])
            
            ########## Buffer ####
            new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
            new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

            new_state = ALIFStateTuple(v=new_v,
                                       z=new_z,
                                       b=new_b,                                
                                       evec_v=new_evec_v, 
                                       evec_a=new_evec_a,   
                                       evec_v_i=new_evec_v_i, 
                                       evec_a_i=new_evec_a_i,  
                                       psi=new_psi_, 
                                       bpsi=new_bpsi_, 
                                       i_future_buffer=new_i_future_buffer,
                                       z_buffer=new_z_buffer)
        return [new_z, new_v, thr, new_psi_, new_bpsi_, e_trace_out,e_trace_i_out, i_syn[:,:,0]], new_state
