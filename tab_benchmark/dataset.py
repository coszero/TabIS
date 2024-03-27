import json
import random
from typing import List
from tab_benchmark.utils import default_load_json


class Dataset():
    def __init__(self, spec_name, shot, serialization, format, metric, verbalizer='none', 
                 url=None, other_info=None, max_data_num=None, raw_dataset=None, shuffle=False, seed=1) -> None:
        self.spec_name = spec_name
        self.shot = shot
        self.serialization = serialization
        self.format = format
        self.metric = metric
        self.other_info = other_info
        self.max_data_num = max_data_num
        self.seed = seed
        self.url = url
        self.shuffle = shuffle

        self.verbalizer = dataset2verb[verbalizer]
        if raw_dataset is None:
            self.raw_dataset = self.download_data(self.url)
        else:
            self.raw_dataset = raw_dataset
        self.dataset = self.post_process_dataset() 

    def download_data(self, url):
        if url.endswith('.json'):
            return default_load_json(url)

    def post_process_dataset(self):
        dataset = []
        if self.format == 'instruction-input-output':
            for rd in self.raw_dataset:
                sample = copy_dict_except_keys(rd, ["instruction", "input", "output"])
                prompt = "instruction:\n" + rd["instruction"] + "\ninput:\n" + rd["input"] + "\noutput:\n"
                output = self.verbalizer(rd["output"])
                sample["input"] = prompt
                sample["output"] = output
                dataset.append(sample)
        elif self.format == 'input-output':
            for rd in self.raw_dataset:
                sample = copy_dict_except_keys(rd, ["input", "output"])
                sample["input"] = rd["input"]
                sample["output"] = self.verbalizer(rd["output"])
                dataset.append(sample)
        elif self.format == 'prompt-response':
            for rd in self.raw_dataset:
                sample = copy_dict_except_keys(rd, ["prompt", "response"])
                sample["input"] = rd["prompt"]
                sample["output"] = self.verbalizer(rd["response"])
                dataset.append(sample)
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(dataset)
        if self.max_data_num is not None:
            if len(dataset) > self.max_data_num:
                print(f"too many test samples: {len(dataset)}, only keep: {self.max_data_num}")
            return dataset[:self.max_data_num]
        else:
            return dataset
    
    @staticmethod
    def from_ad_hoc_dataset(raw_dataset: List[dict], verbalizer='none', metric='', max_data_num=None, shuffle=False, seed=0, suffix=''):
        return Dataset(spec_name=f'ad-hoc-{suffix}', shot='zero', serialization='md', format='prompt-response', metric=metric, verbalizer=verbalizer, 
                max_data_num=max_data_num, raw_dataset=raw_dataset, shuffle=shuffle, seed=seed)

def verbalizer_tabfact(output):
    output = output.lower().strip()
    if output.endswith('.'):
        output = output[:-1]
    if output in ['supported', '1']:
        return 1
    elif output in ['refuted', '0']:
        return 0
    else:
        return output

def verbalizer_list(output):
    try:
        output = json.loads(output)
    except:
        output = []
    output = [str(s).strip() for s in output]
    return output

def verbalizer_formula_reco(output):
    if output.endswith(' * 100'):
        output = output[:-6]
    return output

def verbalizer_cell_lookup(output):
    # if "</s>" in output:
    #     output = output.split("</s>")[0]
    return output.strip()

def verbalizer_table_qa(output):
    output = output.strip()
    if output.endswith('.0'):
        output = output[:-2]
    return output.lower()

def verbalizer_table_qa_mc(output):
    output = output.strip().upper()
    if len(output) == 0:
        return output
    if output[0] in ["A", "B"]:
        return output[0]
    else:
        return output

def verbalizer_none(output):
    return output

dataset2verb = {
            'tabfact': verbalizer_tabfact,
            'value-desc': verbalizer_list,
            'formula-reco': verbalizer_formula_reco,
            'cell-lookup': verbalizer_cell_lookup,
            'row-lookup': verbalizer_list,
            'table-qa': verbalizer_table_qa,
            'table-qa-mc': verbalizer_table_qa_mc,
            'none': verbalizer_none
            }


def is_valid_dataset(dataset_info, shots, serializations, spec_names, datasets):
    shot = dataset_info['shot']
    serialization = dataset_info['serialization']
    # filter by datasets
    if datasets == 'all':
        in_datasets = True
    else:
        dataset_list = datasets.split('/')
        in_datasets = dataset_info["dataset"] in dataset_list
    # filter by spec_name
    if spec_names == 'all':
        in_spec_names = True
    else:
        spec_name_list = spec_names.split('/')
        in_spec_names = dataset_info["spec_name"] in spec_name_list
    # filter by shot and serialization
    in_shots_serializations = shot in shots and serialization in serializations

    if in_datasets and in_spec_names and in_shots_serializations:
        return True
    else:
        return False
    
def copy_dict_except_keys(original_dict, keys_to_exclude):
    return {k: v for k, v in original_dict.items() if k not in keys_to_exclude}