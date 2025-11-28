#!/bin/bash
# CHAIR Evaluation Script for FarSight
# This script runs caption generation using FarSight decoding strategy

# Step 1: Generate captions using FarSight
python eval_chair.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/coco/val2014 \
    --question-file /path/to/chair_questions.jsonl \
    --answers-file ./Answers/chair_captions.jsonl \
    --farsight

# Note: After generating captions, you can evaluate CHAIR metrics using:
# - External CHAIR evaluation tool (e.g., https://github.com/Maxlinn/CHAIR-metric-standalone)
# - Example command:
#   python chair.py --cap_file ./Answers/chair_captions.jsonl \
#       --coco_path /path/to/coco/annotations \
#       --cache ./chair.pkl \
#       --save_path ./Answers/eval-chair.json