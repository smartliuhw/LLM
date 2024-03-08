#! /bin/bash

model_type=Qwen
# 模型选项：chatglm3, Qwen, Baichuan2-13B-4bits, Baichuan2-13B-8bits
port=10086

echo "Starting $model_type on port $port"

# conda activate chatglm

# /home/ubuntu/anaconda3/envs/llm/bin/python /home/ubuntu/AI_exhibition/llm_infer/llm_online_service.py \
python llm_online_service.py \
    --model_type $model_type \
    --port $port <<ESXU
y
y
ESXU
