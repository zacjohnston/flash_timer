#!/bin/bash

# Setup weak-scaling performance tests

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
ln -sf ${PWD}/submit_all_weak.sh ${MODEL_SET}/
ln -sf ${PWD}/submit_omp_batch_weak.sh ${MODEL_SET}/
ln -sf ${PWD}/submit_single.sh ${MODEL_SET}/
ln -sf ${PWD}/submit_job.sb ${MODEL_SET}/

for i in {0..6}; do
	let OMP_THREADS=2**${i}
	let MAX_RANKS=${MAX_MPI}/${OMP_THREADS}
	echo "OMP_THREADS=${OMP_THREADS}"

	for j in {0..12}; do
		let RANKS=2**${j}
		echo "	RANKS=${RANKS}"

		for LEAF_BLOCKS_PER_RANK in {1,2,4,8}; do
			let LEAF_BLOCKS=${RANKS}*${LEAF_BLOCKS_PER_RANK}
			echo "		LEAF_BLOCKS=${LEAF_BLOCKS}"

			MODEL=${MODEL_SET}_${OMP_THREADS}_${LEAF_BLOCKS}_${RANKS}
			MODEL_PATH=${MODEL_SET}/${MODEL}

			mkdir -p ${MODEL_PATH}
			ln -sf ${FLASH_EXE} ${MODEL_PATH}/
			cp flash_weak_${LEAF_BLOCKS}.par ${MODEL_PATH}/flash_${LEAF_BLOCKS}.par
		done

		if [ ${RANKS} -eq ${MAX_RANKS} ]; then
			break
		fi

	done

	if [ ${OMP_THREADS} -eq ${MAX_OMP} ]; then
		break
	fi
done
