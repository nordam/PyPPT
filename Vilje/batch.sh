#! /bin/bash
#PBS -N PyPPT
#PBS -l select=1:ncpus=32:mpiprocs=16
#PBS -l walltime=0:10:00
#PBS -A ntnu065
#PBS -m abe

# Load modules
module load intelcomp/16.0.1
module load mpt/2.14
module load python/2.7.12

# cd to directory where the job was submitted
cd $PBS_O_WORKDIR

# Running job
time -p mpirun -np 16 python ${HOME}/PyPPT/main.py
