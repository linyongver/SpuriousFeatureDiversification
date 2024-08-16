### Introduction
In our ICLR2024 paper [Spurious Feature Diversification Improves Out-of-distribution Generalization](https://openreview.net/forum?id=d6H4RBi7RH), we found that learning diverse spurious features actually improves OOD generalizations, which can be effectively applied to modern (large) DNN. We also have a follow-up work on Large Language works on [Mitigating the Alignment Tax of RLHF](https://arxiv.org/abs/2309.06256) .

### Run the codes
Run the following cmd to reproduce the results on MultiColorMNIST in Table 1 (p=0.7)

`python submit_main.py --seed 1 --id_sp 1.0 --p 0.7 --n_restarts 20 --colors 32`

The arguement
--id_sp is the spurious correlation in the training domain. 
--p is the strength of distributional shift (the spurious correlation in the testing domain is 1-p)



