#!/bin/bash
# Edit a single video with AVE (zero-shot, Wan2.2 I2V backbone).
# Usage:
#   export GEMINI_API_KEY=<your key>
#   bash run_video_edit.sh
set -e

# ============== Configuration ==============
INPUT_VIDEO="./assets/input.mp4"
EDIT_PROMPT="Convert the video to a watercolor style"
CKPT_DIR="./Wan2.2-I2V-A14B"
NUM_GPUS=1          # 1 = single GPU (offloading), >1 = torchrun with FSDP + Ulysses
PORT=7559           # torchrun rendezvous port (multi-GPU only)
WORK_DIR="./outputs/video_edit_workdir"
OUTPUT="./outputs/edited.mp4"
NUM_KEYFRAMES=3     # number of anchor keyframes (incl. first & last)
SEED=42

if [ -z "${GEMINI_API_KEY}" ]; then
    echo "Please set GEMINI_API_KEY first: export GEMINI_API_KEY=<your key>"
    exit 1
fi

# ============== Run ==============
python video_edit.py \
    --input_video "${INPUT_VIDEO}" \
    --edit_prompt "${EDIT_PROMPT}" \
    --ckpt_dir "${CKPT_DIR}" \
    --num_gpus ${NUM_GPUS} \
    --port ${PORT} \
    --work_dir "${WORK_DIR}" \
    --output "${OUTPUT}" \
    --seed ${SEED} \
    --num_keyframes ${NUM_KEYFRAMES}
