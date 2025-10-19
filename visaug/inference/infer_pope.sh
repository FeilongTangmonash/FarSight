python ./visaug/inference/infer_pope.py \
    --model-path /model/llava \
    --question-file ./data/pope/coco/coco_pope_random.json \
    --image-folder ./data/pope/coco/val2014 \
    --answers-file ./outputs/inference/res_coco_random.jsonl \
    --use-visaug \
    --enh-para 1.15 \
    --sup-para 0.95 \
