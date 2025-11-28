#!/bin/bash
# CHAIR Evaluation Script for FarSight
# This script runs caption generation using FarSight decoding strategy

# Configuration - Set these environment variables before running:
# MODEL_PATH: Path to the LLaVA model (default: liuhaotian/llava-v1.5-7b)
# IMAGE_FOLDER: Path to COCO val2014 images
# QUESTION_FILE: Path to CHAIR questions file (JSONL format)
# ANSWERS_FILE: Path to save generated captions (default: ./Answers/chair_captions.jsonl)

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
QUESTION_FILE="${QUESTION_FILE:-}"
ANSWERS_FILE="${ANSWERS_FILE:-./Answers/chair_captions.jsonl}"

# Validate required paths
if [ -z "$IMAGE_FOLDER" ]; then
    echo "Error: IMAGE_FOLDER environment variable is not set"
    echo "Usage: IMAGE_FOLDER=/path/to/coco/val2014 QUESTION_FILE=/path/to/questions.jsonl bash eval_chair.sh"
    exit 1
fi

if [ -z "$QUESTION_FILE" ]; then
    echo "Error: QUESTION_FILE environment variable is not set"
    echo "Usage: IMAGE_FOLDER=/path/to/coco/val2014 QUESTION_FILE=/path/to/questions.jsonl bash eval_chair.sh"
    exit 1
fi

# Create output directory if needed
mkdir -p "$(dirname "$ANSWERS_FILE")"

# Step 1: Generate captions using FarSight
python eval_chair.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$ANSWERS_FILE" \
    --farsight

# Note: After generating captions, you can evaluate CHAIR metrics using:
# - External CHAIR evaluation tool (e.g., https://github.com/Maxlinn/CHAIR-metric-standalone)
# - Example command:
#   python chair.py --cap_file ./Answers/chair_captions.jsonl \
#       --coco_path /path/to/coco/annotations \
#       --cache ./chair.pkl \
#       --save_path ./Answers/eval-chair.json