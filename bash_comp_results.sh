#!/bin/bash

# main comparison table results
# clip & crm results are generated together


dataset=food-101
echo ${dataset}
python main.py --partition test --dataset ${dataset} --prompt clip      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt cupl      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt vcd       --inference vcd  
python main.py --partition test --dataset ${dataset} --prompt hiec      --inference hie   --hie-lambda 0.3
python main.py --partition test --dataset ${dataset} --prompt hiet      --inference hie   --hie-lambda 0.5
python main.py --partition test --dataset ${dataset} --prompt ours-full --inference flat 


dataset=ucf-101
echo ${dataset}
python main.py --partition test --dataset ${dataset} --prompt clip      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt cupl      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt vcd       --inference vcd  
python main.py --partition test --dataset ${dataset} --prompt hiec      --inference hie   --hie-lambda 0.7
python main.py --partition test --dataset ${dataset} --prompt hiet      --inference hie   --hie-lambda 0.3
python main.py --partition test --dataset ${dataset} --prompt ours-full --inference flat 


dataset=cub-200
echo ${dataset}
python main.py --partition test --dataset ${dataset} --prompt clip      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt cupl      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt vcd       --inference vcd  
python main.py --partition test --dataset ${dataset} --prompt hiec      --inference hie   --hie-lambda 0.7
python main.py --partition test --dataset ${dataset} --prompt hiet      --inference hie   --hie-lambda 0.4
python main.py --partition test --dataset ${dataset} --prompt ours-full --inference flat 


dataset=sun-324
echo ${dataset}
python main.py --partition test --dataset ${dataset} --prompt clip      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt cupl      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt vcd       --inference vcd  
python main.py --partition test --dataset ${dataset} --prompt hiec      --inference hie   --hie-lambda 0.6
python main.py --partition test --dataset ${dataset} --prompt hiet      --inference hie   --hie-lambda 0.3
python main.py --partition test --dataset ${dataset} --prompt ours-full --inference flat 


dataset=imagenet
echo ${dataset}
python main.py --partition test --dataset ${dataset} --prompt clip      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt cupl      --inference flat 
python main.py --partition test --dataset ${dataset} --prompt vcd       --inference vcd  
python main.py --partition test --dataset ${dataset} --prompt hiec      --inference hie   --hie-lambda 0.6
python main.py --partition test --dataset ${dataset} --prompt hiet      --inference hie   --hie-lambda 0.3
python main.py --partition test --dataset ${dataset} --prompt ours-full --inference flat 