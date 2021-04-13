#!/bin/bash

if [[ $# -ne 4 ]]; then
  echo -e "\nusage: `basename $0` <model_set> <max_omp> <max_mpi> <node_size>\n"
  echo "where: <model_set> = name of model dir"
  echo "       <max_omp> = max number of OMP threads"
  echo "       <max_mpi> = max number of MPI ranks"
  echo -e "       <node_size> = number of cores on node\n"
  exit 0
fi

MODEL_SET=${1}
MAX_OMP=${2}
MAX_MPI=${3}
NODE_SIZE=${4}

for i in {0..6}; do
	let OMP_THREADS=2**${i}
	let MAX_RANKS=${NODE_SIZE}/${OMP_THREADS}
	echo "===== Submitting batch OMP_THREADS=${OMP_THREADS} ====="

   	for j in {0..12}; do
        	let RANKS=2**${j}
	        echo "  RANKS=${RANKS}"

        	for k in {7,8,9}; do
	                let LEAF_BLOCKS=2**${k}
        	        MODEL=${OMP_THREADS}_${LEAF_BLOCKS}_${RANKS}

                	sbatch --job-name=${MODEL} \
                        	--cpus-per-task=${OMP_THREADS} \
	                        --ntasks=${MAX_RANKS} \
        	                --export=MODEL_SET=${MODEL_SET},LEAF_BLOCKS=${LEAF_BLOCKS},RANKS=${RANKS} \
                	        --output=slurm_logs/slurm_${MODEL_SET}_${MODEL}.output \
                        	submit_job.sb
		done

		if [ ${RANKS} -eq ${MAX_MPI} ]; then
			break
		fi
	done

	if [ ${OMP_THREADS} -eq ${MAX_OMP} ]; then
		break
	fi
done
