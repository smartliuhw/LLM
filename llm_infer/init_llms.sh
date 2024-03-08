#! /bin/bash

model_type=Qwen1.5
# 模型选项：chatglm3, Qwen, Qwen1.5, Baichuan2-13B-4bits, Baichuan2-13B-8bits
port=10086

echo "Starting $model_type on port $port"

# conda activate chatglm

python llm_online_service.py \
    --model_type $model_type \
    --port $port
