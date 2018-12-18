# Three-way balance in the Arctic

This repo contains the scripts used to produce the analysis presented in the paper, as well as the configurations scripts and input files for the idealised modelling. 



To run a simulation with kappa = 0 and 2 cm/s ice speed, invoke the `run_BG.py` script with

```
python run_BG_ice.py -dir u2k0/spin_up  -kappa 0 -wind_mag 0.02
```