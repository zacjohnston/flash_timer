#!/bin/bash

# Setup set of performance tests

if [[ $# -ne 3 ]]; then
  echo -e "\nusage: `basename $0` <type> <model_set> <max_omp>\n"
  echo "where: <type> = scaling test type, either 'strong' or 'weak'"
  echo "       <model_set> = name of model dir"
  echo -e "       <max_omp> = max number of OMP threads\n"
  exit 0
fi


TYPE=${1}
MODEL_SET=${2}
MAX_OMP=${3}

for i in {0..6}; do
	let OMP_THREADS=2**${i}
	echo "===== Submitting batch OMP_THREADS=${OMP_THREADS} ====="
	./submit_omp_batch_${TYPE}.sh ${MODEL_SET} ${OMP_THREADS}

	if [ ${OMP_THREADS} -eq ${MAX_OMP} ]; then
		break
	fi
done
