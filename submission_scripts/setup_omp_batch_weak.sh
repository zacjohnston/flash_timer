#!/bin/bash

# Setup performance tests
#
# Note: must first export OMP_NUM_THREADS as desired

if [[ $# -ne 2 ]]; then
  echo "usage: `basename $0` <model_set> <OMP_NUM_THREADS>"
  exit 0
fi

MODEL_SET=${1}
OMP_NUM_THREADS=${2}

NODE_SIZE=128
let MAX_RANKS=NODE_SIZE/OMP_NUM_THREADS

for i in {0..12}; do 
	let RANKS=2**${i}

	for LEAF_BLOCKS_PER_RANK in {1,2,4,8}; do
		let LEAF_BLOCKS=${RANKS}*${LEAF_BLOCKS_PER_RANK}
		MODEL=${MODEL_SET}_${OMP_NUM_THREADS}_${LEAF_BLOCKS}_${RANKS}

		mkdir -p ${MODEL}/output
		cp -P flash4 ${MODEL}/
		cp flash_${LEAF_BLOCKS}.par ${MODEL}/
	done

	if [ ${RANKS} -eq ${MAX_RANKS} ]; then
		break
	fi
done

