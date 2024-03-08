import sys
from copy import copy
import os
import logging
from sympy import false
import torch
import torch.nn as nn
import numpy as np
import json
import argparse

from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

# from log import get_logger
from utils import load_model_on_gpus

sys.path.append("../")

path_dict = {
    "chatglm3": "/data/liuhanwen/models/chatglm3",
    "Qwen": "/data/AI_exhibition_newest/models/Qwen-7B-hf",
    "Qwen1.5": "/data/liuhanwen/models/Qwen1.5-7B-chat-hf",
    "Baichuan2-13B-4bits": "/data/liuhanwen/models/Baichuan2-13B-chat-hf-4bits",
    "Baichuan2-13B-8bits": "/data/liuhanwen/models/Baichuan2-13B-chat-hf-8bits",
}

Baichuan_user_token = '<reserved_106>'
Baichuan_assistant_token = '<reserved_107>'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--model_type",
                        type=str,
                        default="chatglm3",
                        choices=["chatglm3", "Qwen", "Qwen1.5", "Baichuan2-13B-4bits", "Baichuan2-13B-8bits"]
    )
    args = parser.parse_args()
    return args

# tokenizer = AutoTokenizer.from_pretrained("/data/liuhanwen/models/chatglm3", trust_remote_code=True)
# # model = AutoModel.from_pretrained("/gs/home/liuhwen/models/chatglm3", trust_remote_code=True, device='cuda')
# model = load_model_on_gpus('/data/liuhanwen/models/chatglm3', num_gpus=2)
# model = model.eval()

# In Baichuan tokenizer, the <user_tokens> is <reserved_106>, the <assistant_token> is <reserved_107>

args = parse_args()
model_path = path_dict[args.model_type]
tokenizer = AutoTokenizer.from_pretrained(model_path)
if "chatglm" in args.model_type:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model = model.eval()


app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["GET", "POST", "OPTIONS"])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    # print('headers: ', request.headers)
    # print('content_type: ', request.content_type)
    data = request.json
    # print(data)
    req_param = data["req_param"]
    prompt = req_param["prompt"]
    # print("prompt: ", prompt)
    history = req_param["history"]
    if 'Qwen' == args.model_type and (history is None or history == []):
        history = None
    if "Qwen1.5" == args.model_type and (history is None or history == []):
        history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    print(args.model_type)
    print(history)
    if prompt == "what model":
        return json.dumps(
            {"response": args.model_type, "history": history}, ensure_ascii=False
        )
    # output, token_probs = generate(prompt)
    if "Baichuan" in args.model_type:
        print(history)
        history.append({"role": "user", "content": prompt})
        Baichuan_prompt = history
        # print(Baichuan_prompt)
        response = model.chat(tokenizer, Baichuan_prompt)
        history.append({"role": "assistant", "content": response})
        
    elif "Qwen1.5" in args.model_type:
        print(history)
        history.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        history.append({"role": "assistant", "content": response})
        
    else:
        # print('history: ', history)
        response, history = model.chat(tokenizer, prompt, history=history)

    return json.dumps(
        {"response": response, "history": history}, ensure_ascii=False
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)