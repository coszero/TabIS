import copy
import random
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from functools import partial
from itertools import chain
import time

from datasets_module import Table, TableQASample
from transformers import LlamaTokenizer
from prompt.concat_prompt import get_concat_template, get_qa_template
from data.option.tools import insert_option
from data.tools import aggregate_samples_via_keys, add_id2samples
from tab_benchmark.utils import default_load_json, check_path


class Tokenizer():
    def __init__(self) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", local_files_only=True)

    def compute_tokens(self, text):
        # use llama tokenizer to compute the tokenized length
        tokenized_text = self.tokenizer(text)
        return len(tokenized_text['input_ids'])

def pipeline(tqa_dataset, dataset_info, task_descriptions, evidence, shot, linearization, tf_org_tqas_func, tqas_post_process_func=None, 
             need_gen_option=False, gen_option_func=None, option_dump_path=None,                       # option generation
             gen_opt_strategies=None, keep_opt_num=1, overwrite_cache=False, inverse_option=False,     # option generation
             skip_train=False, max_data_num=None, max_token_length=4000, seed=0, early_return='none'):
    """
    pipeline:
        - 'org' phase: load the dataset from huggingface script/hub and split the dataset
        - 'tqas' phase: transform original sample to TableQASample
        - 'pr' phase: transform TableQASample to prompt-response sample
    
    @param task_descriptions: [dataset_description, multi_choice_description]
    @param early_return: ['none', 'org', 'tqas']
    """
    random.seed(seed)
    assert early_return in ['none', 'org', 'tqas'], f"invalid early_return: {early_return}"
    dataset_description, multi_choice_description = task_descriptions
    if need_gen_option:
        assert multi_choice_description is not None, "must provide description for multi-choice."

    # *load the original dataset from huggingface
    if tqa_dataset.endswith(".json"):
        org_dataset = default_load_json(tqa_dataset)
    else:
        org_dataset = load_dataset(tqa_dataset)
    print(f"successfully load original dataset from {tqa_dataset}")

    # split dataset for evaluation/finetuning
    # dataset keys must be ['train', 'validation', 'test'] or ['train', 'validation']
    assert 'train' in org_dataset.keys(), "origin dataset doesn't have key 'train'"
    assert 'validation' in org_dataset.keys(), "origin dataset doesn't have key 'validation'"
    if 'test' not in org_dataset.keys():
        test_keys = ['validation']
        train_keys = ['train']
    else:
        test_keys = ['test']
        train_keys = ['train', 'validation']
    
    org_train_dataset = aggregate_samples_via_keys(org_dataset, train_keys)
    org_test_dataset = aggregate_samples_via_keys(org_dataset, test_keys)

    # add a unique id to samples
    # TODO: modify from_dataset funcs to enable sample indices.
    org_train_dataset = add_id2samples(org_train_dataset)
    org_test_dataset = add_id2samples(org_test_dataset)
    
    # org_train_dataset = list(org_dataset['train'][x] for x in range(1000))
    # org_test_dataset = list(org_dataset['validation'][x] for x in range(1000))

    # org_train_dataset = [org_dataset[k] for k in train_keys]
    # # optimize: when not need to chain
    # if len(org_train_dataset) > 1:
    #     org_train_dataset = list(chain(*org_train_dataset))
    # else:
    #     org_train_dataset = list(org_train_dataset[0])
    # org_test_dataset = [org_dataset[k] for k in test_keys]
    # if len(org_test_dataset) > 1:
    #     org_test_dataset = list(chain(*org_test_dataset))
    # else:
    #     org_test_dataset = list(org_test_dataset[0])

    print(f"split: train: {len(org_train_dataset)}, test: {len(org_test_dataset)}")

    # clip datasets. following max_data_num, skip_train and shot.
    random.shuffle(org_train_dataset)
    random.shuffle(org_test_dataset)
    if max_data_num is not None:
        print(f"for the training set and test set, keep {max_data_num} at most.")
        org_train_dataset = org_train_dataset[:max_data_num]
        org_test_dataset = org_test_dataset[:max_data_num]
        org_test_dataset = org_test_dataset[:10]
    if skip_train:
        # if skip_train is True, only keep few-shot demonstrations
        assert max_data_num >= shot
        org_train_dataset = org_train_dataset[:(shot+10)]  # in case that the samples are too long
    if early_return == 'org':
        return org_train_dataset, org_test_dataset
    
    # *transfer the original dataset to TableQA dataset
    tqa_sample_train = [tf_org_tqas_func(d) for d in org_train_dataset]
    tqa_sample_test = [tf_org_tqas_func(d) for d in org_test_dataset]
    tqa_sample_train = [d for d in tqa_sample_train if d is not None]
    tqa_sample_test = [d for d in tqa_sample_test if d is not None]
    
    # TableQASample post-processing
    if tqas_post_process_func is not None:
        tqa_sample_train = tqas_post_process_func(tqa_sample_train)
        tqa_sample_test = tqas_post_process_func(tqa_sample_test)

    # generate no-option demonstrations for option generation. generate_k_shot_template
    assert shot >= 1 and shot <=len(tqa_sample_train), f"get shot = {shot}, len(tqa_sample_train) = {len(tqa_sample_train)}"
    tqa_sample_train = sorted(tqa_sample_train, key=lambda x: len(x.table.md_table))
    opt_demo_tqa_samples = tqa_sample_train[:shot]
    tqa_sample_train = tqa_sample_train[shot:]
    if "options" in evidence:
        no_opt_evidence = [e for e in evidence if e != "options"]
        no_opt_demo_qa = generate_demo_qa_template(opt_demo_tqa_samples, no_opt_evidence, linearization)
    else:
        no_opt_demo_qa = generate_demo_qa_template(opt_demo_tqa_samples, evidence, linearization)
    # add options to TableQASample
    if need_gen_option:
        assert gen_option_func is not None
        gen_option_func = partial(gen_option_func, 
                                demo_qa=no_opt_demo_qa,
                                gen_opt_strategies=gen_opt_strategies,
                                evidence=evidence,
                                task_description=dataset_description,
                                linearization=linearization,
                                option_dump_path=option_dump_path,
                                keep_opt_num=keep_opt_num,
                                max_token_length=max_token_length,
                                overwrite_cache=overwrite_cache,
                                inverse_option=inverse_option
                                )
        tqa_sample_train = gen_option_func(tqa_sample_train, data_type="train")
        tqa_sample_test = gen_option_func(tqa_sample_test, data_type="test")
        # TODO: add option source.
        check_path(option_dump_path)
        TableQASample.dump_check_option_md(tqa_sample_test, option_dump_path)

    if early_return == 'tqas':
        return tqa_sample_train, tqa_sample_test

    # *transfer the TableQA sample to prompt-response sample
    assert shot >= 1 and shot <=len(tqa_sample_train), f"get shot = {shot}, len(tqa_sample_train) = {len(tqa_sample_train)}"
    demo_tqa_samples = tqa_sample_train[:shot]
    tqa_sample_train = tqa_sample_train[shot:]
    # demo_qa = generate_demo_qa_template(demo_tqa_samples, evidence, linearization)
    demo_qa = ['Question: Based on the following table, which statement about 2010 and 417 is accurate?\nMeta Information:\npage_title: Harrod, Ohio, section_title: Demographics\nTable:\n| Historical population | Historical population | Historical population | Historical population |\n| --- | --- | --- | --- |\n| Census | Pop. |  | %± |\n| 1890 | 269 |  | — |\n| 1900 | 370 |  | 37.5% |\n| 1910 | 474 |  | 28.1% |\n| 1920 | 389 |  | −17.9% |\n| 1930 | 421 |  | 8.2% |\n| 1940 | 422 |  | 0.2% |\n| 1950 | 482 |  | 14.2% |\n| 1960 | 563 |  | 16.8% |\n| 1970 | 533 |  | −5.3% |\n| 1980 | 506 |  | −5.1% |\n| 1990 | 537 |  | 6.1% |\n| 2000 | 491 |  | −8.6% |\n| 2010 | 417 |  | −15.1% |\n| Est. 2017 | 402 |  | −3.6% |\n| U.S. Decennial Census | U.S. Decennial Census | U.S. Decennial Census | U.S. Decennial Census |\nOptions:\nA. As of the census of 2010, there were 417 people, residing in the Harrod.\nB. In 2010, the historical population in Harrod, Ohio was 417.\nAnswer: A']
    if need_gen_option:
        task_description = multi_choice_description
    else:
        task_description = dataset_description
    tf_tqas_pr_func = partial(transform_tqas_dataset_to_pr_dataset, dataset_info=dataset_info,
                              evidence=evidence, task_description=task_description, 
                              linearization=linearization, max_token_length=max_token_length)
    if not skip_train:
        pr_sample_train = tf_tqas_pr_func(tqas_dataset=tqa_sample_train, demo_qa=demo_qa)
    else:
        pr_sample_train = []
    pr_sample_test = tf_tqas_pr_func(tqas_dataset=tqa_sample_test, demo_qa=demo_qa)
    return pr_sample_train, pr_sample_test

def generate_demo_qa_template(demo_tqa_samples, evidence, linearization, sep_qa=False):
    demos = [fill_qa_template(s, evidence, linearization, is_demo=True, sep_qa=sep_qa) for s in demo_tqa_samples]
    return demos
    
def fill_qa_template(tqa_sample: TableQASample, evidence: List[str], linearization: str, is_demo=False, sep_qa=False):
    """
    Fill in QA templates. The qa_template is automatically detected based on (1) evidence and (2) contained of the evidence.
    If one kind of evidence is empty, it won't appear in the final prompt.
    """

    assert linearization in ["md", "html"]
    if linearization == "md":
        table_md = Table.convert_table_data_to_md_str(tqa_sample.table.grid_table)
    elif linearization == "html":
        table_md = Table.convert_table_data_to_html_str(tqa_sample.table.table_data)
    
    answer = tqa_sample.answer if is_demo else ''
    if sep_qa:
        map_dict = {"question": tqa_sample.question, "table": table_md}
        ans_map_dict = {"answer": answer}
    else:
        map_dict = {"question": tqa_sample.question, "table": table_md, "answer": answer}

    # add caption, passage, options to map_dict
    has_caption = False
    has_passage = False
    has_options = False
    if "caption" in evidence:
        caption = tqa_sample.table.meta["summary"]
        if caption:      #TODO: more complex inspection. 
            map_dict.update({"caption": caption})
            has_caption = True
    if "passage" in evidence:
        passage = tqa_sample.passage
        if passage:
            map_dict.update({"passage": passage})
            has_passage = True
    if "options" in evidence:
        has_options = True
        options = tqa_sample.options
        if isinstance(options, list) and len(options) > 0:
            raise ValueError("when concatenating the prompt, the options has to be str.")
        map_dict.update({"options": options})

    if sep_qa:
        q_template, a_template = get_qa_template(sep_qa=True, has_caption=has_caption, has_passage=has_passage, has_options=has_options)
        return q_template.format_map(map_dict), a_template.format_map(ans_map_dict)
    else:
        qa_template = get_qa_template(sep_qa=False, has_caption=has_caption, has_passage=has_passage, has_options=has_options)
        return qa_template.format_map(map_dict)

def transform_tqas_dataset_to_pr_dataset(tqas_dataset: List['TableQASample'], dataset_info, demo_qa, evidence, 
                                         task_description, linearization, max_token_length=4000):
    """
    Transform a TableQASample dataset to a prompt-response dataset
    - fill in templates (involves table linearization)
    - filter too long samples

    @param demo_qa: a list of demonstrations
    """
    concat_template = get_concat_template()
    pr_dataset = []
    tokenizer = Tokenizer()
    for data in tqdm(tqas_dataset):
        input_q = fill_qa_template(data, evidence, linearization, is_demo=False, sep_qa=False)
        # concat demos
        concat_demo_qa = '\n\n'.join(demo_qa)
        prompt = concat_template.format_map({"task_description": task_description, "demo_qa": concat_demo_qa, "input_q": input_q})
        response = str(data.answer)
        token_length = tokenizer.compute_tokens(prompt)
        if token_length <= max_token_length:
            sample = dict(prompt=prompt, response=response)
            sample.update(dataset_info)
            sample.update({"options": data.options, "option_types": data.option_types})
            if data.other_info != None:
                sample.update(data.other_info)
            pr_dataset.append(sample)
    print(f"using max_token_length={max_token_length}, sample num: {len(pr_dataset)}, keeping {round(len(pr_dataset)/ len(tqas_dataset)*100, 2)}% samples")
    return pr_dataset


def transform_tqa_dataset_to_zero_shot_iio_dataset(tqas_dataset: List['TableQASample'], evidence, demo_qa,
                                                    task_description, linearization, max_token_length=4000):
    """
    Transform a TableQASample dataset to a zero-shot instruction-input-output dataset
    - v1: this function is mainly used for option generation
    - v2 update: add `demo` for one-shot prompting; `demo` is a concatenation of input-output pairs.
                 For some tasks, i.g. examing in exam-judge, one-shot prompting is neccessary.
    """

    def get_md_table_str(tqa_sample: 'TableQASample', linearization):
        if linearization == "md":
            table_md = Table.convert_table_data_to_md_str(tqa_sample.table.grid_table)
        elif linearization == "html":
            table_md = Table.convert_table_data_to_html_str(tqa_sample.table.table_data)
        else:
            raise ValueError(f"invalid linearization: {linearization}")
        return table_md
    
    if "options" in evidence:
        print("For iio samples, `options` are not expected in evidence. Remove options from the evidence.")
        evidence = copy.deepcopy(evidence)
        evidence.remove("options")

    # When the sep_qa is True, this function outputs the assembled question and answer separately. 
    # And they are used as input and output, respectively
    samps = generate_demo_qa_template(tqas_dataset,
                                       evidence=evidence,
                                       linearization=linearization,
                                       sep_qa=True)
    md_tables = [get_md_table_str(tqa_sample, linearization) for tqa_sample in tqas_dataset]
    meta_infos = [tqa_sample.table.meta["summary"] for tqa_sample in tqas_dataset]
    highlighted_cells = [tqa_sample.other_info.get("highlighted_cells", []) for tqa_sample in tqas_dataset]
    iio_dataset = []
    # tokenizer = Tokenizer()
    for (question, answer), meta_info, md_table, hcells in zip(samps, meta_infos, md_tables, highlighted_cells):
        # token_num = tokenizer.compute_tokens(task_description + "\n" + question)
        # no need to limit max_token_length
        # even result in bugs in generator.py
        # if token_num <= max_token_length:
        iio_dataset.append(dict(instruction=task_description, input=question, output=answer, demo=demo_qa,
                                meta_info=meta_info, md_table=md_table, highlighted_cells=hcells))
    return iio_dataset


