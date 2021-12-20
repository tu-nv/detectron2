#!/bin/bash
# run this from bash
# LOSS_TYPES="giou diou ciou jgiou jdiou jciou alpha_iou cdiou" ./train.sh

set -e
set -x

cd "$(dirname "$0")"

loss_types=($LOSS_TYPES)

CONFIG_FILE="voc_losses_faster_rcnn_R_50_FPN"

for LOSS_TYPE in "${loss_types[@]}"; do
    ./train_net.py \
        --config-file ../configs/PascalVOC-Detection/${CONFIG_FILE}.yaml \
        --num-gpus 1 \
        --dist-url "auto" \
        MODEL.RPN.BBOX_REG_LOSS_TYPE "$LOSS_TYPE" \
        MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE "$LOSS_TYPE" \
        OUTPUT_DIR "./output/$LOSS_TYPE-${CONFIG_FILE}"
done
