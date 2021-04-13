#!/bin/bash

# Submit performance tests

if [[ $# -ne 5 ]]; then
  echo -e "\nusage: `basename $0` <model_set> <omp_threads> <leaf_blocks> <mpi_ranks> <node_size>\n"
  exit 0
fi

MODEL_SET=${1}
OMP_THREADS=${2}
LEAF_BLOCKS=${3}
RANKS=${4}
NODE_SIZE=${5}

let MAX_RANKS=${NODE_SIZE}/${OMP_THREADS}
MODEL=${OMP_THREADS}_${LEAF_BLOCKS}_${RANKS}

sbatch --job-name=${MODEL} \
	--cpus-per-task=${OMP_THREADS} \
	--ntasks=${MAX_RANKS} \
	--export=MODEL_SET=${MODEL_SET},LEAF_BLOCKS=${LEAF_BLOCKS},RANKS=${RANKS} \
	--output=slurm_logs/slurm_${MODEL_SET}_${MODEL}.output \
	submit_job.sb
