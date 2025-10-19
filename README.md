# Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding
<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://mllms-farsight.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
## Abstract
Recent advancements in multimodal large language models (MLLMs) have significantly improved performance in visual question answering. However, they often suffer from hallucinations. In this work, hallucinations are categorized into two main types: initial hallucinations and snowball hallucinations. We argue that adequate contextual information can be extracted directly from the token interaction process. Inspired by causal inference in decoding strategy, we propose to leverage causal masks to establish information propagation between multimodal tokens. The hypothesis is that insufficient interaction between those tokens may lead the model to rely on outlier tokens, overlooking dense and rich contextual cues. Therefore, we propose to intervene in the propagation process by tackling outlier tokens to enhance in-context inference. With this goal, we present FarSight, a versatile plug-and-play decoding strategy to reduce attention interference from outlier tokens merely by optimizing the causal mask. The heart of our method is effective token propagation. We design an attention register structure within the upper triangular matrix of the causal mask, dynamically allocating attention capture attention diverted to outlier tokens.
Moreover, a positional awareness encoding method with a diminishing masking rate is proposed, allowing the model to attend to further preceding tokens, especially for video sequence tasks. With extensive experiments, FarSight demonstrates significant hallucination-mitigating performance across different MLLMs on both image and video benchmarks, proving its effectiveness.

<div style="text-align: center;">
    <img src="Fig/intro.png" alt="Example image" style="width:65%; height:auto;">
</div>


## Implementation
<div style="text-align: center;">
    <img src="Fig/method.png" alt="Example image" style="width:65%; height:auto;">
</div>

## Evaluation

- [x] Setup
- [x] Visual Neglect in Modal Fusion
- [x] VAF Inference & Evaluation

---

#### Setup

```powershell
conda create -n farsight python=3.10
conda activate farsight
cd LLaVA
pip install -e .
```

---

#### Visual Neglect in Modal Fusion

We provide the following script to reproduce our analysis results on the over-reliance of multimodal large language models on linguistic priors.

```powershell
bash ./visaug/analysis/vis_flow.sh
```

or

```bash
python ./visaug/analysis/vis_flow.py \
    --model-path /model/llava \
    --question-file ./data/pope/coco/coco_pope_random.json \
    --image-folder ./data/pope/coco/val2014 \
    --answers-file ./outputs/analysis/res_coco_random.pt 
```

The analysis results are shown in the two figures below, from which we can draw two key conclusions:

- The model performs the crucial fusion of visual and textual modalities in the middle layers, creating cross-modal semantic representations that drive the final predictions.

- During this critical fusion process, the model demonstrates inadequate attention to the visual modality.

<img title="" src="images/2024-12-18-15-57-55-image.png" alt="" data-align="center" width="625">

---



## Farsight Inference & Evaluation

###  POPE Evaluation

Use the following scripts to reproduce the experimental results on the **POPE** benchmark.

```bash
bash ./visaug/inference/infer_pope.sh
bash ./visaug/inference/eval_pope.sh
```

---

###  CHAIR Evaluation

Use the following script to evaluate the model on the **CHAIR** benchmark.

```bash
bash ./visaug/inference/eval_chair.sh
```

---


## Acknowledgement

## Citation
```
@inproceedings{tang2025seeing,
  title={Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding},
  author={Tang, Feilong and Liu, Chengzhi and Xu, Zhongxing and Hu, Ming and Huang, Zile and Xue, Haochen and Chen, Ziyang and Peng, Zelin and Yang, Zhiwei and Zhou, Sijin and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26147--26159},
  year={2025}
}
```