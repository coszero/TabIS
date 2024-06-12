

import os
import copy
import subprocess
from tab_benchmark.chatapi import get_reply, get_reply_completion, get_reply_from_api
from tab_benchmark.utils import default_dump_json, default_load_json, check_path, default_load_jsonl, get_logger

logger = get_logger(__name__)

class BaseChat(object):
    """
    Base chat: chat with ChatGPT, maintain a dialog, and save the log
    """
    COMPLETION_MODE_MODELS = ["gpt-3.5-turbo-instruct", "text-davinci-002", "text-davinci-003"]
    def __init__(self, model="gpt-3.5-turbo", sys_prompt="", url=None) -> None:
        self.sys_prompt = sys_prompt
        self.model = model
        self.url = url
        self.use_completion = model in BaseChat.COMPLETION_MODE_MODELS
        self.history = [{'role': 'system', 'content': sys_prompt}]

    
    def chat(self, prompt, maintain_history=True, temperature=0, n=1, verbose=False):
        # maintain_history only available for chat mode
        if self.use_completion:
            maintain_history = False

        if self.url is None:
            # By default, the openai does not require url
            if self.use_completion:
                reply = get_reply_completion(prompt=prompt, model=self.model, temperature=temperature, n=n)
            else:
                reply, history = get_reply(prompt=prompt, history=copy.deepcopy(self.history), 
                                        model=self.model, temperature=temperature, n=n)
        else:
            reply = get_reply_from_api(prompt=prompt, url=self.url, model=self.model, n=n, temperature=temperature)
            history = dict(prompt=prompt, response=reply)
        if verbose:
            print(reply)
        if maintain_history:
            self.history = history  
        return reply
    
    def log(self, log_fn_path):
        default_dump_json(log_fn_path, self.history)


class APIAgent():
    NAME = "APIAgent"
    def __init__(self, spec_name, model, url=None, sys_prompt="", temperature=0, n=1) -> None:
        self.spec_name = spec_name
        self.temperature = temperature
        self.model = model
        self.url = url
        self.sys_prompt = sys_prompt
        self.n = n  

    def forward(self, sample, verbalizer=None, with_indices=False, dump_dir=None):
        if with_indices:
            idx, sample = sample
        gpt = BaseChat(model=self.model, url=self.url, sys_prompt=self.sys_prompt)
        response = gpt.chat(sample["input"], maintain_history=True, temperature=self.temperature, n=self.n, verbose=False)
        if verbalizer:
            response = verbalizer(response)
        reply_info = dict(response=response, label=sample["output"], sample=sample, history=gpt.history)
        if dump_dir is not None:
            dump_path = os.path.join(dump_dir, f"{idx}.json")
            default_dump_json(reply_info, dump_path)
            # print(f"sucessfully save: {idx}")
        return reply_info
    
    @staticmethod
    def from_setting(spec_name, setting) -> 'APIAgent':
        print(setting)
        return APIAgent(spec_name=spec_name, model=setting['model'], url=setting.get("url"), sys_prompt=setting['sys_prompt'])
    

class OpenLLMAgent():
    NAME = "openllm"
    def __init__(self, spec_name, setting) -> None:
        self.spec_name = spec_name
        self.setting = setting
    
    def forward_all_samples(self, samples, gpu_devices, script_path, max_memory_per_device, verbalizer=None, dump_dir=None):
        # (1) dump the samples & dataset_info.json to dump dir; (2) run the prediction script from LLaMa-Efficient-Tuning project; 
        # (3) load and process the prediction results
        
        data_path = os.path.join(dump_dir, 'test_data.json')
        default_dump_json(samples, data_path)

        command = f"""python {script_path} \
        --model_name {self.spec_name} \
        --model_path {self.setting["model_path"]} \
        --dataset_json_path {data_path} \
        --dump_path {dump_dir} \
        --visible_devices {gpu_devices} \
        --max_memory_per_device {max_memory_per_device} \
        """
        subprocess.run(command, shell=True)

        model_pred_res = default_load_json(os.path.join(dump_dir, 'pred_res.json'))
        return model_pred_res


    @staticmethod
    def from_setting(spec_name, setting) -> 'OpenLLMAgent':
        logger.info(f"{spec_name}: {setting}")
        return OpenLLMAgent(
            spec_name=spec_name,
            setting=setting
        )


available_models = [APIAgent, OpenLLMAgent]
model_name2class = {model_class.NAME: model_class for model_class in available_models}