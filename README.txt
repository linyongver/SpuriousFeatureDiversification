Run the following cmd to reproduce the results in Table 1 (p=0.7)


python submit_main.py --seed 1 --id_sp 1.0 --p 0.7 --n_restarts 20 --colors 32


Here

--id_sp is the spurious correlation in the training domain. 
--p is the strength of distributional shift (the spurious correlation in the testing domain is 1-p)
