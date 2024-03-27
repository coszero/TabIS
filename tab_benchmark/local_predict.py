import os
from transformers import (LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForCausalLM, AutoTokenizer, AutoConfig, MistralForCausalLM)
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import torch
import json
from tqdm import tqdm
import argparse


def load_dataset_json(dataset_json_path):
    with open(dataset_json_path, 'r') as f:
        return json.loads(f.read())

def load_model_tokenizer_from_path(model_name, model_path, max_memory):
    # if model_name == "llama2-13b-chat":
    if "llama" in model_name or "structlm" in model_name or "tulu" in model_name:
        no_split_module_classes = LlamaForCausalLM._no_split_modules
        # This config is for llama2-13b bfloat16 on L40S,
        # Please change this config if using other devices or modle to
        # ensure your model can be loaded and inference properly.
        config = LlamaConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes) 
        load_checkpoint_in_model(model,model_path, device_map=device_map)
        model = dispatch_model(model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        torch.set_grad_enabled(False)
        model.eval()
    elif model_name == "mistral-7b":
        no_split_module_classes = MistralForCausalLM._no_split_modules
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
        load_checkpoint_in_model(model, model_path, device_map=device_map)
        model = dispatch_model(model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        torch.set_grad_enabled(False)
        model.eval()
    return model, tokenizer

def convert_prob_to_answer(prob_pair):
    if prob_pair[0] >= prob_pair[1]:
        return "A"
    else:
        return "B"

def predict(model_name, model_path, dataset_json_path, max_memory):
    
    data_json = load_dataset_json(dataset_json_path)
    model, tokenizer = load_model_tokenizer_from_path(model_name, model_path, max_memory)

    # get token_id of 'A' and 'B'
    id_A = tokenizer('A', return_tensors="pt").input_ids[0][1]
    id_B = tokenizer('B', return_tensors="pt").input_ids[0][1]
    print(f"get id_A:{id_A}, id_B:{id_B}")
    
    # inference one by one
    pred_res = []
    with tqdm(total=len(data_json)) as t:
        for sample in data_json:
            # torch.cuda.empty_cache()
            if "llama" in model_name or "structlm" in model_name:
                ids = tokenizer(sample['input'], padding=True, truncation=True, return_tensors="pt")
                ids = ids.to(model.device)
                ids = {k : v for k, v in ids.items() if k != "token_type_ids"}
                outputs = model(**ids)
            elif model_name == "mistral-7b":
                messages = [
                    {"role": "user", "content": sample['input']},
                ]
                input = tokenizer.apply_chat_template(messages, return_tensors="pt")
                input = input.to(model.device)
                outputs = model(input)
            softmax_outputs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            response = convert_prob_to_answer(softmax_outputs[-1, [id_A, id_B]].tolist())
            pred_res.append(dict(response=response, label=sample["output"], sample=sample))
            t.update(1)
    return pred_res

def infer_max_memory(visible_devices, max_memory_per_device):
    devices = visible_devices.split(",")
    max_memory_ = f"{max_memory_per_device}GiB"
    return {int(device): max_memory_ for device in devices}



if __name__ == '__main__':
    '''
    Please check Your config.
    Large model can be loaded dispersedly.
    max_memory = {
        4: '15GiB',
        6: '25GiB',
    }
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama", 
                        help="Please specifiy the name of the model.")
    parser.add_argument("--model_path", type=str,
                        help="Please specifiy the model for evaluations.")
    parser.add_argument("--dataset_json_path", type=str,
                        help="Please specifiy the md-one data for evaluations.")
    parser.add_argument("--dump_path", type=str, default=None, 
                        help="Please specifiy the path to dump results.")
    parser.add_argument("--visible_devices", type=str, default="4,5,6", 
                        help="Please specifiy the cuda_available_devices, split by ,.")
    parser.add_argument("--max_memory_per_device", type=int, default="15", 
                        help="Please specifiy the maximum memory per device.")
    args = parser.parse_args()
    print(args)

    max_memory = infer_max_memory(args.visible_devices, args.max_memory_per_device)
    print(max_memory)
    pred_res = predict(args.model_name, args.model_path, args.dataset_json_path, max_memory)
    json.dump(pred_res, open(os.path.join(args.dump_path, "pred_res.json"), 'w'))