#!/bin/bash

# input argument hiec or hiet for respective HIE variant's language prompt
prompt=$1
clear


dataset=food-101
echo ${dataset}
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.1 
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.2
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.3
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.4
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.5
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.6
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.7
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.8
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.9


dataset=ucf-101
echo ${dataset}
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.1 
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.2
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.3
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.4
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.5
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.6
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.7
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.8
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.9


dataset=cub-200
echo ${dataset}
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.1 
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.2
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.3
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.4
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.5
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.6
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.7
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.8
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.9


dataset=sun-324
echo ${dataset}
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.1 
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.2
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.3
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.4
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.5
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.6
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.7
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.8
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.9


dataset=imagenet
echo ${dataset}
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.1 
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.2
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.3
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.4
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.5
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.6
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.7
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.8
python main.py --partition val --prompt ${prompt} --inference hie  --dataset ${dataset}  --hie-lambda 0.9







