import os
import json
import sys
import requests

def check_model():
    data = {"req_param": {"prompt": "what model", "history": None}}
    res = requests.post(url, json=data).json()
    response, history = res["response"], res["history"]
    return response

def init_history(model_type):
    if "chatglm" in model_type:
        history = []
    elif "Qwen" in model_type:
        history = None
    elif "Baichuan" in model_type:
        history = []
    else:
        assert "Wrong model type!"
    return history


url = "http://10.134.102.68:10086/chat"
print('Type \"clean\" to clean the chat history\nType \"exit\" to exit the program')
model_type = check_model()
history = init_history(model_type)

while True:
    prompt = input("prompt: ")
    if prompt == "clean":
        history = init_history(model_type)
        continue
    if prompt == "exit":
        break
    data = {"req_param": {"prompt": prompt, "history": history}}

    res = requests.post(url, json=data).json()
    response, history = res["response"], res["history"]
    print("response: ", response)
    # print("history: ", history)

print("exit")
