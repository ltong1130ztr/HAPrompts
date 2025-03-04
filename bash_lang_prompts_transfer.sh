#!/bin/bash

# evaluation of our language prompts transferability across different LLMs

# CLIP & CRM as reference
python main.py --partition test --dataset imagenet --inference flat --prompt clip             


# ours-full + gpt-3.5-turbo-0125
python main.py --partition test --dataset imagenet --inference flat --prompt ours-full        


# ours-full + claude-3.5-sonnet
python main.py --partition test --dataset imagenet --inference flat --prompt ours-full-claude 


# ours-full + gemini-1.5-flash
python main.py --partition test --dataset imagenet --inference flat --prompt ours-full-gemini 