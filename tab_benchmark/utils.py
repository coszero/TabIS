import os
import re
import yaml
import json
import pickle
import jsonlines
import logging


def default_load_jsonl(jsonl_path, encoding="utf-8", max_row=None):
    data = []
    with open(jsonl_path, 'r', encoding=encoding) as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
            if max_row is not None and len(data) > max_row:
                break
    return data

def default_dump_jsonl(content, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        for row in content:
            writer.write(row)

def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json

def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)
        
def default_load_yaml(file_path):
    return yaml.load(open(file_path, encoding='utf8'), Loader=yaml.FullLoader)

def default_dump_md(md_files, dump_path):
    fp = os.path.join(dump_path)
    with open(fp, 'w') as file:
        for s in md_files:
            file.write(s + '\n')

def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj

def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)

def default_load_txt(file_path):
    rows = []
    with open(file_path, 'r') as f:
        for row in f:
            rows.append(row)
    return rows

def add_dump_txt(text, file_path):
    with open(file_path, 'a') as f:
        f.write(f"{text}\n")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
