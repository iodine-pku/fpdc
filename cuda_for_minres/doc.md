# Requirements:
nvcc=11.0.221
gcc=9.4.0
Python=3.7.12
numpy=1.21.6
scipy=1.7.3
pycuda=2022.1
Driver Version: 470.82.01
CUDA Version: 11.4


# Files:
\src:
cpu_minres.py and gpu_minres_v3.py, gpu_minres_v4.py are source scripts
experiments.ipynb: contains some interactive experiments used to generate figures and tables for the report(not very readable).
analysis.ipynb: draw graphs for the report

\report:
contains report.pdf and report source

\pics:
contains images for the report

# Scripts:
To run the script at local machine, use sh run.sh
To submit it to server, use run.slurm

There is no compile script since this is automatically done by python.

