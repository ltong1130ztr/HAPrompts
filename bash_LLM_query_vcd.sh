#!/bin/bash

# generate VCD image prompts

python gen_vcd_chatgpt_prompts.py --dataset food-101
python gen_vcd_chatgpt_prompts.py --dataset ucf-101
python gen_vcd_chatgpt_prompts.py --dataset cub-200
python gen_vcd_chatgpt_prompts.py --dataset sun-324
python gen_vcd_chatgpt_prompts.py --dataset imagenet