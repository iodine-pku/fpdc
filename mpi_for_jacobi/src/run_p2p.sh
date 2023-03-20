#!/bin/bash
module add mpich
sbatch run1.slurm
sleep 5
sbatch run4.slurm
sleep 5
sbatch run16.slurm
sleep 5