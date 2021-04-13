#!/bin/bash

if [[ $# -ne 3 ]]; then
  echo -e "\nusage: `basename $0` <model_set> <max_omp> <node_size>\n"
  echo "where: <model_set> = name of model dir"
  echo "       <max_omp> = max number of OMP threads"
  echo -e "       <node_size> = number of cores on node\n"
  exit 0
fi

MODEL_SET=${1}
MAX_OMP=${2}
NODE_SIZE=${3}

for i in {0..6}; do
	let OMP_THREADS=2**${i}
	echo "===== Submitting batch OMP_THREADS=${OMP_THREADS} ====="
	./submit_omp_batch_strong.sh ${MODEL_SET} ${OMP_THREADS} ${NODE_SIZE}

	if [ ${OMP_THREADS} -eq ${MAX_OMP} ]; then
		break
	fi
done
