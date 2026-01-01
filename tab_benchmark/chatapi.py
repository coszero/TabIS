import os
import httpx
from fastapi import FastAPI, Form
from tab_benchmark.utils import add_dump_txt, check_path
from httpx_socks import SyncProxyTransport, AsyncProxyTransport
import time
import datetime
import json
import random
import copy

openai_chat_api = "https://api.openai.com/v1/chat/completions"
openai_completion_api = "https://api.openai.com/v1/completions"
openai_emb_api = "https://api.openai.com/v1/embeddings"
openai_key = os.getenv("OPENAI_KEY")

open_router_api = "https://openrouter.ai/api/v1/chat/completions"
open_router_key = os.getenv(
    "OPENROUTER_KEY",
    "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",
)

headers_openai = {
    "Authorization": f"Bearer {openai_key}"
}
headers_open_router = {
    "Authorization": f"Bearer {open_router_key}"
}
headers_app = {'Content-Type': 'application/json'}

# better log path
log_base_path = os.path.join(os.getenv("HOME"), "openai_cost/")
if not os.path.exists(log_base_path):
    os.makedirs(log_base_path)
dump_to = f"{log_base_path}/{{}}.txt"

def get_reply(prompt, history=None, model = "gpt-3.5-turbo", temperature=0, n=1):
    """
    available models: ['gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613',
                        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613']
    """
    assert n >= 1
    # transport = SyncProxyTransport.from_url(proxy)
    while True:
        try:
            new_history = copy.deepcopy(history)
            if new_history is None:
                new_history = []
            new_history.append({"role": "user", "content": prompt})
            with httpx.Client() as client:
                resp = client.post(open_router_api, json={"model": model,
                                                          "messages": new_history, "temperature": temperature, "n": n,
                                                          }, headers=headers_open_router, timeout=5*60)
                data = resp.json()
                if data.get('choices'):
                    usage = data['usage']
                    cost = compute_cost(usage, model)
                    add_dump_txt(cost, dump_to.format(str(datetime.date.today())))
                    if n > 1:
                        reply = [c['message'] for c in data['choices']]
                        # new_history.append({"role": "assistant", "content": reply[0]})
                        new_history.append(reply)
                    else:
                        reply = data['choices'][0]['message']['content']
                        new_history.append({"role": "assistant", "content": reply})
                    return reply, new_history
                else:
                    raise ValueError(f"return: {data}")
        except Exception as e:
            # new_history = copy.deepcopy(history)
            # new_history.append({"role": "user", "content": prompt})
            # new_history.append({"role": "assistant", "content": ""})
            # return "", new_history
            print(e)
            print("sleep and retry...")
            time.sleep(10)

def get_reply_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, n=1):
    """
    available models: ['gpt-3.5-turbo-instruct', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']
    """
    assert n >= 1
    transport = SyncProxyTransport.from_url(proxy)
    while True:
        try:
            with httpx.Client(transport=transport) as client:
                resp = client.post(open_router_api, json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "n": n
                }, headers=headers_open_router, timeout=2*60)
                data = resp.json()
                if data.get('choices'):
                    usage = data['usage']
                    cost = compute_cost(usage, model)
                    add_dump_txt(cost, dump_to.format(str(datetime.date.today())))
                    if n > 1:
                        reply = [c['message']['content'] for c in data['choices']]
                    else:
                        reply = data['choices'][0]["message"]["content"]
                    return reply
                else:
                    raise ValueError(f"return: {data}")
        except Exception as e:
            print(e)
            print("sleep and retry...")
            time.sleep(10)

def get_reply_from_api(prompt, url, model, headers=headers_app, n=1, **kwargs):
    assert n >= 1
    if model.startswith("accounts/fireworks"):
        headers = headers_fire
    # get packing method
    packing_func = get_packing_func(model)
    # get parsing method
    parse_result_func = get_parse_result_func(model)
    if model == "gemini-pro":  # for gemini-pro, there are 
        url = random.sample(url, 1)[0]
    while True:
        try:
            if "gemini" in model:
                # some models may have to use the proxy.
                http_client = httpx.Client(transport=SyncProxyTransport.from_url(proxy))
            else:
                http_client = httpx.Client()
            # send a message
            with http_client as client:
                data = packing_func(prompt, n)
                if "gemini" not in model:
                    # gemini does not use extra parameters
                    data.update(kwargs)
                resp = client.post(url, json=data, headers=headers, timeout=3 * 60)
                data = resp.json()
                if "error" in data:
                    raise ValueError(f"error: {data}")
                else:
                    reply = parse_result_func(data)
                    if reply == "[INVALID]":
                        print(data)
                    return reply
        except Exception as e:
            if "gemini" in model:
                # must wait for API s with rate limit.
                print(e)
                print("sleep and retry...")
                time.sleep(10)
            else:
                return ""

def get_parse_result_func(model):
    # if model == "chatdoc-mistralai" or model.startswith("accounts/fireworks"):
    if model.startswith("gpt"):
        return lambda data: data
    elif "gemini" in model:
        return lambda data: data["candidates"][0]["content"]["parts"][0]["text"] \
                            if "candidates" in data and "content" in data["candidates"][0] \
                            else "[INVALID]"
    else:
        return lambda data: data["choices"][0]["message"]["content"] if len(data["choices"]) == 1 else [c["message"] for c in data["choices"]]

def get_packing_func(model):
    if "gemini" in model:
        cats = ["HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        safety_setting = [{"category": cat, "threshold": "BLOCK_NONE"} for cat in cats]
        return lambda x, n: {
                    "contents": [{"parts":[{"text":x}]}],
                    "safetySettings": safety_setting,
                    "generationConfig": {
                        "temperature": 0,
                        "n": n
                    }
                }
    else:
        return lambda x, n: {
                        "model": model,
                        "messages": [{"role": "user", "content": x}],
                        "n": n
                        }

def get_embedding(input_text, model="text-embedding-3-large"):
    transport = SyncProxyTransport.from_url(proxy)
    trial = 0
    while trial < 3:
        try:
            with httpx.Client(transport=transport) as client:
                resp = client.post(openai_emb_api, json={"model": model, "input": input_text}, 
                                        headers=headers_openai, timeout=1 * 60)
                data = resp.json()
                if data.get('data'):
                    reply = data['data'][0]['embedding']
                    usage = data['usage']
                    cost = compute_cost(usage, model)
                    add_dump_txt(cost, dump_to.format(str(datetime.date.today())))
                    return reply
                else:
                    return []
                    raise ValueError(f"return: {data}")
        except Exception as e:
            trial += 1
            # print(e)
            # print("sleep and retry...")
            time.sleep(5)
    return []

def get_model_price(model):
        if model.startswith("gpt-4"):
            if "gpt-4-1106" in model:
                return [0.01/1000, 0.03/1000]
            elif "32k" in model:
                return [0.06/1000, 0.12/1000]
            else:
                return [0.03/1000, 0.06/1000]
        if model.startswith("gpt-3.5-turbo"):
            if "instruct" in model:
                return [0.0015/1000, 0.002/1000]
            else:
                return [0.0005/1000, 0.0015/1000]
        if model in ["text-davinci-002", "text-davinci-003"]:
            return [0.02/1000, 0.02/1000]
        if model == "text-embedding-ada-002":
            return [0.0001/1000, 0]
        elif model == "text-embedding-3-large":
            return [0.00013/1000, 0]
        elif model == "text-embedding-3-small":
            return [0.00002/1000, 0]
        
        return [0, 0]


def compute_cost(usage, model):
    input_token = usage['prompt_tokens']
    output_token = usage.get('completion_tokens', 0)
    input_price, output_price = get_model_price(model)
    cost = input_token*input_price + output_token*output_price
    return cost

def report_cost(date=None):
    """
    @param date: str. format: 2023-07-18
    """
    if date is None:
        date = str(datetime.date.today())

    log_path = dump_to.format(date)
    if not os.path.exists(log_path):
        dollar = 0
    else:
        costs = []
        with open(dump_to.format(date), 'r') as file:
            for s in file:
                if s != '\n':
                    costs.append(float(s.strip()))
        dollar = sum(costs)
    print(f"date: {date}")
    print(f"cost: {round(dollar, 4)} dollars.")
