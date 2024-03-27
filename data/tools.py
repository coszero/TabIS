import json
import re
import os
import random
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from data.option.tools import format_list
from tab_benchmark.utils import check_path, default_dump_json, default_load_json
from sklearn.model_selection import train_test_split

def add_id2samples(dataset):
    for sidx, sample in enumerate(dataset):
        sample["id"] = str(sidx)
    return dataset

def aggregate_samples_via_keys(org_dataset, keys):
    flat_dataset = []
    for key in keys:
        print(f"loading samples from {key}...")
        for sample in tqdm(org_dataset[key]):
            flat_dataset.append(sample)
            if len(flat_dataset) >= 100000:
                break
    return flat_dataset

def extract_bracket_content_by_json(s, key = 'answers'):
    json_s = re.findall('\{.*?\}', s)[0]
    try:
        result = json.loads(json_s)[key]
        result = [format_list(x) for x in result]
    except:
        print(f"json.loads error when loading: {s}")
        result = [""]
    return result

def extract_json_content(s):
    s = s.replace("\n", "")
    json_s = re.findall('\{.*?\}', s)
    return json_s[0]

def common_generate_meta_summary(meta_list):
    # concat non-empty meta information
    # meta_list specifies the concatenation order.
    meta_strs = []
    for meta_key, content in meta_list:
        if content != '':
            meta_str = meta_key + ': ' + content
            meta_strs.append(meta_str)
    return ', '.join(meta_strs)

def common_generate_passage(passage_list):
    # concat non-empty passages
    # passage_list specifies the concatenation order.
    passage_strs = []
    for pkey, passage in passage_list:
        if passage != '':
            passage_str = pkey + ': ' + passage
            passage_strs.append(passage_str)
    return '\n'.join(passage_strs)

def train_test_split_sample(samples: List, test_size=0.2):
    random.seed(0)
    # y = [0 for _ in range(len(samples))]
    # X_train, X_test, y_train, y_test = train_test_split(samples, y, test_size=test_size, random_state=0)
    test_num = int(len(samples)*test_size)
    # print(len(samples), test_num)
    assert test_num > 0, "test_num <= 0"
    # for s in samples[-5:]:
    #     print(s["question"])
    test_samples = random.sample(samples, test_num)
    # print(test_samples[0])
    train_samples = [s for s in samples if s not in test_samples]
    return train_samples, test_samples

def save_benchmark_data(data, dump_base_path, task_name, dataset_name, 
                        shot='zero', linearization='md', suffix=''):
    task_path = os.path.join(dump_base_path, "benchmark", task_name, dataset_name)
    check_path(task_path)
    file_name = f"{linearization}-{shot}-{suffix}.json" if suffix else f"{linearization}-{shot}.json"
    file_path = os.path.join(task_path, file_name)
    default_dump_json(data, file_path)
    print(f"successfully dump to {file_path}")

def save_finetune_data(data, dump_base_path, task_name, dataset_name, 
                        shot='zero', linearization='md', suffix=''):
    file_name = f"{task_name}-{dataset_name}-{linearization}-{shot}-{suffix}.json" if suffix \
        else f"{task_name}-{dataset_name}-{linearization}-{shot}.json"
    base_path = os.path.join(dump_base_path, "train")
    check_path(base_path)
    file_path = os.path.join(base_path, file_name)
    default_dump_json(data, file_path)

    dataset_info_path = os.path.join(base_path, "dataset_info.json")
    spec_name = f"{dataset_name}-{linearization}-{shot}"
    this_dataset_info = {
        spec_name: 
        {"file_name": file_name,
        "columns": {
        "prompt": "prompt",
        "query": "",
        "response": "response",
        "history": ""
        }
        }
    }
    if os.path.exists(dataset_info_path):
        dataset_info = default_load_json(dataset_info_path)
    else:
        dataset_info = {}
    dataset_info.update(this_dataset_info)
    default_dump_json(dataset_info, dataset_info_path)
    print(f"successfully dump to {file_path}")