#!/bin/bash

# ensemble method comparison: ensemble over class text embedding space vs ensemble over sub-classes' log-probability (logit) space

# ensemble over log-probability space + ours-full language prompts
inference=vcd
python main.py --partition val --dataset food-101 --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset ucf-101  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset cub-200  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset sun-324  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset imagenet --inference ${inference} --prompt ours-full 

# ensemble over class text embedding space + ours-full language prompts (Our approach)
inference=flat
python main.py --partition val --dataset food-101 --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset ucf-101  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset cub-200  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset sun-324  --inference ${inference} --prompt ours-full 
python main.py --partition val --dataset imagenet --inference ${inference} --prompt ours-full 