#!/bin/bash

# generate our image prompts for 
# (1) main comparison results 
# (2) ablation studies
# (3) language prompts transferability results

# (1) & (2) -------------------------------------
dataset=food-101
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-path
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-comp
python merge.py --dataset ${dataset} --prompt ours-full
# for ablation study
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-LP
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-AP


dataset=ucf-101
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-path
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-comp
python merge.py --dataset ${dataset} --prompt ours-full
# for ablation study
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-LP
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-AP


dataset=cub-200
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-path
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-comp
python merge.py --dataset ${dataset} --prompt ours-full
# for ablation study
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-LP
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-AP


dataset=sun-324
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-path
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-comp
python merge.py --dataset ${dataset} --prompt ours-full
# for ablation study
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-LP
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-AP


dataset=imagenet
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-path
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-comp
python merge.py --dataset ${dataset} --prompt ours-full
# for ablation study
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-LP
python gen_ours_chatgpt_prompts.py  --dataset ${dataset} --prompt ours-AP


# (3) ------------------------------------------- 
# claude image prompts for imagenet-1k
python gen_ours_claude_prompts.py --prompt ours-comp-claude
python gen_ours_claude_prompts.py --prompt ours-path-claude
python merge.py --dataset imagenet --prompt ours-full-claude


# gemini image prompts for imagenet-1k
python gen_ours_gemini_prompts.py --prompt ours-comp-gemini
python gen_ours_gemini_prompts.py --prompt ours-path-gemini
python merge.py --dataset imagenet --prompt ours-full-gemini

