#!/bin/bash
clear


dataset=food-101
echo ${dataset}
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-LP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-AP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-comp
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-AP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-LP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-full


dataset=ucf-101
echo ${dataset}
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-LP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-AP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-comp
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-AP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-LP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-full


dataset=cub-200
echo ${dataset}
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-LP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-AP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-comp
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-AP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-LP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-full


dataset=sun-324
echo ${dataset}
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-LP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-AP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-comp
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-AP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-LP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-full


dataset=imagenet
echo ${dataset}
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-LP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-AP
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-comp
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-AP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt merge --merging-prompts ours-LP ours-path
python main.py --partition val --dataset ${dataset} --inference flat  --prompt ours-full

