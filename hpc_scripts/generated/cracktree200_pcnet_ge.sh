#!/bin/bash
#PBS -l select=2:ncpus=10:mem=60gb
#PBS -l walltime=60:00:00
#PBS -q common_cpuQ
#PBS -N cracktree200_pcnet_ge
#PBS -M erik.nielsen@unitn.it
#PBS -m abe

cd ${PBS_O_WORKDIR}

module load python-3.10.14
module load cuda-11.3
source $PWD/.venv/bin/activate

python3 $PWD/src/main.py $PWD/src/config/hpc/cracktree200/pcnet_ge.json $PBS_ARRAY_INDEX
