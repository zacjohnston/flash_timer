#!/bin/bash

# Submit performance tests

if [[ $# -ne 3 ]]; then
  echo -e "\nusage: `basename $0` <model_set> <OMP_THREADS> <node_size>\n"
  exit 0
fi

MODEL_SET=${1}
OMP_THREADS=${2}
NODE_SIZE=${3}

let MAX_RANKS=${NODE_SIZE}/${OMP_THREADS}

for i in {0..12}; do 
	let RANKS=2**${i}

	for j in {0,1,2,3}; do
#		let LEAF_BLOCKS=${MAX_RANKS}*2**${j}
		let LEAF_BLOCKS=128*2**${j}
		MODEL=${OMP_THREADS}_${LEAF_BLOCKS}_${RANKS}

		sbatch --job-name=${MODEL} \
			--cpus-per-task=${OMP_THREADS} \
			--ntasks=${MAX_RANKS} \
			--export=MODEL_SET=${MODEL_SET},LEAF_BLOCKS=${LEAF_BLOCKS},RANKS=${RANKS} \
			--output=slurm_logs/slurm_${MODEL_SET}_${MODEL}.output \
			submit_job.sb
	done

	if [ ${RANKS} -eq ${MAX_RANKS} ]; then
		break
	fi
done

