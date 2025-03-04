#!/bin/bash

# generate HIE image prompts 

# HieC (original HIE approach with k-means clustering on the fly)
method=hiec
python gen_hie_chatgpt_prompts.py --method ${method} --dataset food-101
python gen_hie_chatgpt_prompts.py --method ${method} --dataset ucf-101
python gen_hie_chatgpt_prompts.py --method ${method} --dataset cub-200
python gen_hie_chatgpt_prompts.py --method ${method} --dataset sun-324
python gen_hie_chatgpt_prompts.py --method ${method} --dataset imagenet

# HieT (our alternative variant with the given label hierarchy replacing k-means clustering results)
method=hiet
python gen_hie_chatgpt_prompts.py --method ${method} --dataset food-101
python gen_hie_chatgpt_prompts.py --method ${method} --dataset ucf-101
python gen_hie_chatgpt_prompts.py --method ${method} --dataset cub-200
python gen_hie_chatgpt_prompts.py --method ${method} --dataset sun-324
python gen_hie_chatgpt_prompts.py --method ${method} --dataset imagenet