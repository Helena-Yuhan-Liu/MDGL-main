# MDGL-main

Code adapted by Yuhan Helena Liu, PhD Candidate, University of Washington

Code adapted for Liu et al., PNAS, 2021 [1]

Code built on top of https://github.com/IGITUGraz/LSNN-official [2]

Please refer to LICENSE for copyright details

## Multidigraph learning (MDGL)

This repository provides a tensorflow 1.12 library and a tutorial to train a recurrent spiking neural networks (obeying Dale's law and connection sparsity constraints) using multidigraph learning rule (MDGL) detailed in our paper [1]. MDGL is the SOTA biologically plausible learning rule inspired by the widespread intracortical neuromodulation illuminated in the recent genetic data from the Allen Institute [3]. MDGL builds on top of the great work of RFLO by Murray [4] and e-prop by Bellec et al. [5]. 

The code is built on top of the LSNN code published for [2] with these important additions:

1. Computing the local modulatory signals that encode the contribution of each neuron activity z to the loss 
2. Computing and accumulating eligibility trace
3. Combine the previous steps for weight updates 
4. Save the data and plot the learning curves 

## Usage

In the folder `bin/` there is the code for the evidence accumulation task (Figure 3C in [1]). The saved data for Figure 3C is already included and you can load them using 'plot_results.py'. You can generate data from five more runs (of BPTT, e-prop and MDGL) by running 'sh run_AccuCues.sh default' (may take a few days) and visualize the saved data using 'plot_results.py'. In the folder `lsnn/` you may find the source code of the lsnn package retained from [2].

Instead of modifying the gradient using @tf.custom_gradient and exploiting Tensorflow's automatic differentiation, this code explicitly computes all quantities involved in MDGL (i.e. eligibility trace and modulatory learning signals), in order to have them available for analysis. A consequence of this is that this code is not memory optimized and requires 60G (it explicitly stores large tensors, e.g. eligibility trace tensor for recurrent weights has a shape of B x T x N x N, where B=64, T=2000, N=100 for Fig 3C [1]). 

## Installation

The installation instruction is copied from (https://github.com/IGITUGraz/LSNN-official). The code is compatible with python 3.4 to 3.7 and tensorflow 1.7 to 1.12 (CPU and GPU versions).

> You can run the training scripts **without installation** by temporarily including the repo directory
> in your python path like so: `` PYTHONPATH=. python3 bin/tutorial_sequential_mnist_with_LSNN.py`` 

From the main folder run:  
`` pip3 install --user .``  
You can now import the tensorflow cell called ALIF (for adaptive leakey integrate and fire) as well as the rewiring wrapper to update connectivity matrices after each call to the optimizer.
Warning, the GPU compatible version of tensorflow is not part of the requirements by default.
To use GPUs one should also install it:
 ``pip3 install --user tensorflow-gpu``.

## References:

[1] Yuhan Helena Liu, Stephen Smith, Stefan Mihalas, Eric Shea-Brown, and Uygar Sümbül. Cell-type-specific neuromodulation guides synaptic credit assignment in a spiking neural network. PNAS (accepted), 2021.

[2] Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. “Long short-term memory and learning-to-learn in networks of spiking neurons”. In: 32nd Conference on Neural Information Processing Systems. 2018, pp. 787–797.

[3] Stephen J. Smith, Uygar Sümbül, Lucas T. Graybuck, Forrest Collman, Sharmishtaa Seshamani, Rohan Gala, Olga Gliko, Leila Elabbady, Jeremy A. Miller, Trygve E. Bakken, Jean Rossier, Zizhen Yao, Ed Lein, Hongkui Zeng, Bosiljka Tasic, and Michael Hawrylycz. Single-cell transcriptomic evidence for dense intracortical neuropeptide networks. eLife, 8, nov 2019.

[4] J. M. Murray, Local online learning in recurrent networks with random feedback.
eLife 8, e43299 (2019).

[5] Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11(1):1–15, dec 2020.
