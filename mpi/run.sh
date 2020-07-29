#!/bin/bash
#SBATCH -J run-mixmodel
#SBATCH -o log-run.o%j
#SBATCH -e log-run.e%j
#SBATCH -N 4
#SBATCH -t 24:00:00
#SBATCH -p cca
# --constraint=skylake

source ~/.bash_profile
# init_conda
# conda activate pymc3
init_env

cd /mnt/ceph/users/apricewhelan/projects/cuddly-system/scripts

date

# export KMP_INIT_AT_FORK=False
# export OPENBLAS_NUM_THREADS=1
# export MKL_THREADING_LAYER=TBB

mpirun python3 run-mixmodel.py --mpi

date

