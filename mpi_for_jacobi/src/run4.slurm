#!/bin/bash
#SBATCH -o ./out/job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH -t 5:00

module add mpich
mpiexec -n 4 ./bin/p2p.exe
