**This is the code for paper "# Transferable Adversarial Attacks on SAM and Its Downstream Models"**

This file reproduces the experimental results on attacking the medical SAM. Most codes for training and evaluating the [Medical SAM](https://github.com/hitachinsk/SAMed) comes from SAMed by [hitachinsk](https://github.com/hitachinsk).

1. To conduct our codes, pls download the SAM pretrained [model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints), and Lora [fine-tuned parameters](https://drive.google.com/file/d/1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr/view) for medical segmentation. Put them in the `./checkpoints` folder.
2. Please download the [evaluation set](https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link)
3. To conduct the UMI-GRAT and evaluate other methods, please simply run

```python
python  run_attacks.py
```

Please modify the hyperparameters in the** run_attacks.py** documents for utilization. As the meta-initialization learning process is time-consuming, we suggest using the UMI provided by us.

You try different basic attack strategies and combine them with ours by adjusting "methods = ['MI-FGSM']" and "UMI_GRs=[1]" in **run_attacks.py**


If you find this code helpful, please consider citing:

```
@inproceedings{
xia2024transferable,
title={Transferable Adversarial Attacks on {SAM} and Its Downstream Models},
author={Song Xia and Wenhan Yang and Yi Yu and Xun Lin and Henghui Ding and LINGYU DUAN and Xudong Jiang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=yDjojeIWO9}
}
```
