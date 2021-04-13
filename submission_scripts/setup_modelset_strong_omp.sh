#!/bin/bash

# Setup set of performance tests

if [[ $# -ne 4 ]]; then
  echo -e "\nusage: `basename $0` <model_set> <exe_basename> <max_omp> <max_mpi>\n"
  echo "where: <model_set> = name of model dir"
  echo "       <exe_basename> = basename of flash4 model, e.g. 'sod3d_amd' for '\${BANG}/obj/obj_sod3d_amd/flash4'"
  echo "       <max_omp> = max number of OMP threads (must be power of 2)"
  echo -e "       <max_mpi> = max number of MPI ranks to use (must be power of 2)\n"
  exit 0
fi


MODEL_SET=${1}
EXE_BASENAME=${2}
MAX_OMP=${3}
MAX_MPI=${4}

FLASH_EXE=${BANG}/obj/obj_${EXE_BASENAME}/flash4

PWD=$(pwd)
mkdir -p ${MODEL_SET}/slurm_logs
ln -sf ${PWD}/submit_all_strong_omp.sh ${MODEL_SET}/
ln -sf ${PWD}/submit_single.sh ${MODEL_SET}/
ln -sf ${PWD}/submit_job.sb ${MODEL_SET}/

for i in {0..6}; do
	let OMP_THREADS=2**${i}
	echo "OMP_THREADS=${OMP_THREADS}"

	for j in {0..12}; do
		let RANKS=2**${j}
		echo "	RANKS=${RANKS}"

		for k in {7,8,9}; do
			let LEAF_BLOCKS=2**${k}
			if [ ${j} -eq 0 ]; then
				echo "		LEAF_BLOCKS=${LEAF_BLOCKS}"
			fi

			MODEL=${MODEL_SET}_${OMP_THREADS}_${LEAF_BLOCKS}_${RANKS}
			MODEL_PATH=${MODEL_SET}/${MODEL}

			mkdir -p ${MODEL_PATH}
			ln -sf ${FLASH_EXE} ${MODEL_PATH}/
			cp flash_strong_${LEAF_BLOCKS}.par ${MODEL_PATH}/flash_${LEAF_BLOCKS}.par
		done

		if [ ${RANKS} -eq ${MAX_MPI} ]; then
			break
		fi

	done

	if [ ${OMP_THREADS} -eq ${MAX_OMP} ]; then
		break
	fi
done
