# Scalable Thompson Sampling via Optimal Transport

The **particle-interactive Thompson sampling** (\pi-TS) uses distribution optimization techniques to approximate the posterior distribution is Thompson sampling, solved via Wasserstein gradient flows. Our approach is scalable and does not make explicit distribution assumptions on posterior approximations. This repository contains source code to reproduce the results presented in the paper [Scalable Thompson Sampling via Optimal Transport](https://users.cs.duke.edu/~ryzhang/Ruiyi/OT_TS.pdf) (AISTATS 2019):

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
2. [Experimental Codes](#experiments)
3. [Reproduce paper figure results](#reproduce-paper-figure-results) 

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.5.0](https://www.tensorflow.org/) and [Theano==0.9.0]


## Acknowledgement

This implementation is based on [Deep Bayesian bandits library](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits). We thank for Carlos Riquelme et al. for making their code public.
