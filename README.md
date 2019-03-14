# Scalable Thompson Sampling via Optimal Transport

The **Particle-Interactive Thompson sampling** (\pi-TS) uses distribution optimization techniques to approximate the posterior distribution in Thompson sampling, solved via Wasserstein gradient flows. Our approach is scalable and does not make explicit distribution assumptions on posterior approximations. This repository contains source code to reproduce the results presented in the paper [Scalable Thompson Sampling via Optimal Transport](https://users.cs.duke.edu/~ryzhang/Ruiyi/OT_TS.pdf) (AISTATS 2019):

```
@inproceedings{Zhang_pi_TS,
  title={Scalable Thompson Sampling via Optimal Transport},
  author={Ruiyi Zhang, Zheng Wen, Changyou Chen, Chen Fang, Tong Yu, Lawrence Carin},
  booktitle={AISTATS},
  year={2019}
}
```

## Contents
We provides the codes of proposed methods and produicing figures: 
1. [Dependencies](#dependencies)
2. [Experimental Codes](#Experimental-Codes)

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.5.0](https://www.tensorflow.org/) and [Theano==0.9.0](http://deeplearning.net/software/theano/)

## Experimental Codes
### Dataset ###
Download the required datasets with the following cmd:
```
python prepare_data.py
```

### Training ###

Train the model on the data.
```
source $dataset$.sh
```

Training log is printed as below:
```
Initializing model BBB-bnn.
Initializing model NeuralLinear-bnn.
Initializing model DGF-bnn.
Successfully initialized the models!
Initializing model BootRMS-0-bnn.
Initializing model BootRMS-1-bnn.
Initializing model BootRMS-2-bnn.
Training SVGD-bnn for 100 steps...
Training BBB-bnn for 100 steps...
Training NeuralLinear-bnn for 100 steps...
Training DGF-bnn for 100 steps...
Training BootRMS-0-bnn for 100 steps...
Training BootRMS-1-bnn for 100 steps...
Training BootRMS-2-bnn for 100 steps...
...

```

## Evaluation

The results will be saved in a \*.npz file, and the figures are based on these files.

## Acknowledgement

This implementation is based on [Deep Bayesian bandits library](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits). We thank Riquelme et al. for making their code public.
