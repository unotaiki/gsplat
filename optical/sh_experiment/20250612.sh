# bin/bash

# this is the experiment of refractive gaussian splatting

if mcmc; then
    --scale-reg 0.01

python3 optical/simple_trainer_refraction.py \
    mcmc \
    --
    --result-dir "results/mcmc_" \