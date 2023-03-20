#!/bin/bash
module add mpich
sbatch run_1.slurm
sleep 5
sbatch run_4.slurm
sleep 5
sbatch run_16.slurm
sleep 5