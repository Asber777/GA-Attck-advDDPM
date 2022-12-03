#!/bin/bash
CUDA_ID="0,1,2"
TARGET_M="Standard_R50"
SOURCE_ID_list="2_3_5"
Auxiliary_id_list="1_4_6"
Attack_method="GA_DMI_FSA"
LOSS="margin"
MAX_EPSILON=3.5
EPSILON=3.5
Num_steps=50
Momentum=1.0
batch_size=10
THRES=0.3
Intervals=5
kernel_size=5
mode="nearest"
Feature=$1 
PORT=10164
NPROC=3

OUTPUT_DIR="Output_Feature_APR/Ensemble_TID_${TARGET_M}_SIDLIST_${SOURCE_ID_list}_Auxiliary_id_list_${Auxiliary_id_list}_${Attack_method}_${LOSS}_thres_${THRES}_intervals_${Intervals}_steps_${Num_steps}_max_eps_${MAX_EPSILON}_eps_${EPSILON}_mu_${Momentum}_mode_${mode}_kernel_size_${kernel_size}_${Feature}"
mkdir -p ${OUTPUT_DIR}

echo ${OUTPUT_DIR}

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python -u main_ens_attack.py \
  --save_path=${OUTPUT_DIR}/images \
  --attack_method=${Attack_method} \
  --loss_fn=${LOSS} \
  --batch_size=${batch_size} \
  --max_epsilon=${MAX_EPSILON} \
  --epsilon=${EPSILON} \
  --num_steps=${Num_steps} \
  --intervals=${Intervals} \
  --momentum=${Momentum} \
  --thres=${THRES} \
  --mode=${mode} \
  --kernel_size=${kernel_size} \
  --source_list=${SOURCE_ID_list} \
  --auxiliary_list=${Auxiliary_id_list} \
  --target_m=${TARGET_M}