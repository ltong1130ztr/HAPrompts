#!/bin/bash

# generate PyTorch ViT inference results on ImageNet-1k

python pretrained_vit.py --model ViT-B-16
python pretrained_vit.py --model ViT-B-32
python pretrained_vit.py --model ViT-L-16
python pretrained_vit.py --model ViT-L-32