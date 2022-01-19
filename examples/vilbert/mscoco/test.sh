#!/bin/bash

TASK=7
MODEL=vilbert
MODEL_CONFIG=vilbert_base
TASKS_CONFIG=vilbert_test_tasks
PRETRAINED=/home/gushertni@GU.GU.SE/aicsproject/volta/AgwrMiOjTv
OUTPUT_DIR=results/mscoco/${MODEL}

source activate volta

cd ../../..
python eval_retrieval.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test --batch_size 1 \
	--output_dir ${OUTPUT_DIR}

conda deactivate
