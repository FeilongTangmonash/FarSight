import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

#from AttnAdapter import AttnAdapter
from PIL import Image
import math
from farsight_patch import FarSightContext

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def fname_to_cocoid(name: str) -> int:
    base = os.path.splitext(os.path.basename(name))[0]   # '000000002323'
    return int(base.lstrip('0') or '0')                  # 全0时返回 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--farsight", action="store_true", help="enable FarSight decoding")
    parser.add_argument("--farsight-decay", type=float, default=None, help="FarSight sigma; default=log(256)/log(1024)")
    parser.add_argument("--farsight-no-alibi", action="store_true", help="disable ALiBi-like head slopes")

    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    fs_ctx = (
        FarSightContext(
            model,
            decay_factor=args.farsight_decay,
            use_alibi=not args.farsight_no_alibi,
        )
        if args.farsight else None
    )

        # 读入问题 + 结果文件
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(q) for q in f]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    ctx_mgr = fs_ctx if fs_ctx is not None else torch.no_grad()
    with ctx_mgr:
        for line in tqdm(questions):
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            image_id_val = fname_to_cocoid(image_file)

            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # 重要：FarSight 版本未实现 KV cache，建议 use_cache=False
            # （不影响正确性，只是推理略慢；后续可升级支持 KV cache）
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=1024,
                use_cache=(not args.farsight),  # 开 FarSight 时关掉 cache
            )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "caption": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "image_id": image_id_val,
                "metadata": {}
            }) + "\n")
            ans_file.flush()
    ans_file.close()
