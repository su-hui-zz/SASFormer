#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi

export PYTHONPATH="$PWD":$PYTHONPATH
#DATA_ROOT="$PWD/data"
DATA_ROOT="/data1"
DATA_DIR="${DATA_ROOT}/"
#SAVE_DIR="${DATA_ROOT}/Seg_result/VOC2012"
SAVE_DIR="./Seg_result/VOC2012"

BACKBONE="mixtransformer"

CONFIGS="configs/voc2012/Segformerb4_512_2.json"

MODEL_NAME="segformer"
LOSS_TYPE="fs_ce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_b4_512_2_aff1.2"$2
LOG_FILE="./log/voc2012/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_DIR="./pretrained/"

MAX_ITERS=80000
SIGMA=0.02

if [ "$1"x == "train"x ]; then
                      python3 -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_DIR} \
                       --nbb_mult 10.0 \
                       --sigma ${SIGMA} \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  python3 -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1 2 3 \
                       --resume_continue y \
                       --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --sigma ${SIGMA} \
                       2>&1 | tee -a ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  python3 -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --val_batch_size 1 \
                       --resume ./checkpoints/voc2012/${CHECKPOINTS_NAME}_max_performance.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image --sigma ${SIGMA} \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val

  cd lib/metrics
  python3 -u pascal_context_evaluator.py --configs ${CONFIGS} \
                                         --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val/label \
                                         --gt_dir ${DATA_DIR}/val/label

else
  echo "$1"x" is invalid..."
fi