import os
import argparse
import inspect
from functools import partial

import sys
sys.path.append('.')
# print(sys.path)
from tab_benchmark.utils import default_load_yaml
from data.tools import save_benchmark_data, save_finetune_data
from data import datasets_funcs
from data.data_converter import pipeline
from config.paths import get_paths

DATACONFIGPATH = "./config/data_config_generation.yaml"
BASESCRIPTPATH = "./data/raw_dataset"

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        help="Please specifiy the dataset.")
    parser.add_argument("--shot", type=int, default=1,
                        help="k-shot. k >= 1")
    parser.add_argument("--linearization", type=str,
                        help="the strategy to linearize the table. Current avialable: md, html")
    parser.add_argument("--gen_opt_strategies", default=["exam_judge"], nargs="+",
                        help="the strategy to generate options")
    parser.add_argument("--suffix", type=str, default='',
                        help="the suffix after the saved files. May be used to distinguish different experiments.")
    parser.add_argument("--max_token_length", type=int, default=4000,
                        help="the max token length of the prompt and response")
    parser.add_argument("--max_data_num", type=int, default=-1,
                        help="the maximum data number of the training set and the test set, respectively")
    parser.add_argument("--keep_opt_num", type=int, default=1,
                        help="the number of options")
    parser.add_argument("--dataset_config_path", type=str, default=DATACONFIGPATH)
    parser.add_argument("--base_script_path", type=str, default=BASESCRIPTPATH)
    parser.add_argument("--machine", default='c10', type=str,
                        help="which machine this script runs on")
    parser.add_argument("--skip_train", action='store_true',
                        help='whether to generate finetuning data.')
    parser.add_argument("--overwrite_cache", action='store_true',
                        help='whether to overwrite the cache of option generation.')
    parser.add_argument("--inverse_option", action='store_true',
                        help='whether to generate a quiz with the inversed options at the same time.')
    args = parser.parse_args()
    return args

def parse_dataset_config(config_path, base_script_path, raw_json_path):
    # generate the dict: dataset_name to dataset_config
    # the script path is constructed by concatenating the base_script_path / raw_json_path and 
    # the script name (.py or .json or using huggingface datasets) 
    dataset_config = default_load_yaml(config_path)
    data_name2config = {}
    for task_type, datasets in dataset_config.items():
        for data_name, data_config in datasets.items():
            if "script_name" in data_config:   # multi-choice may not contain `script_name`
                script_name = data_config["script_name"]
                if script_name.endswith(".py"):
                    data_config["script_path"] = os.path.join(base_script_path, script_name)
                elif script_name.endswith(".json"):
                    data_config["script_path"] = os.path.join(raw_json_path, script_name)
                else:
                    # using huggingface datasets
                    data_config["script_path"] = script_name
            data_config["task_type"] = task_type
            data_name2config[data_name] = data_config
    return data_name2config

def get_dataset_func(func_name):
    available_funcs = [name for name, obj in inspect.getmembers(datasets_funcs) if inspect.isfunction(obj)]
    func = getattr(datasets_funcs, func_name, None)
    assert func is not None, f"""invalid func: {func_name}. available funcs: {available_funcs}"""
    return func

def generate_and_save(datasets, data_name2config, option_base_path, dump_to, shot, linearization, suffix, 
                      max_token_length, gen_opt_strategies, keep_opt_num, inverse_option=False,
                      max_data_num=None, skip_train=False, overwrite_cache=False):
    
    for dataset in datasets:
        print(f"generating {dataset}...")
        config = data_name2config[dataset]
        script_path = config.get("script_path", None)
        org2tqas_func = config.get("org2tqas_func", None)
        post_process_func = config.get("post_process_func", None)
        dataset_info = dict(task_type=config["task_type"], dataset=dataset)
        # handle gen_option_func
        need_gen_option = dataset.startswith("mc-")
        dataset_dir = f"{dataset}-{suffix}/" if suffix else f"{dataset}/"
        option_dump_path = os.path.join(option_base_path, dataset_dir) if need_gen_option else None
        if need_gen_option:
            # this will inherit following attrs from the original setting.
            #   1. `task_description`. [Required] Used as `dataset description`
            #   2. `script_path`, `org2tqas_func` and `post_process_func` [Optional] only inherit if not specified.
            assert "org_dataset" in config and config["org_dataset"] in data_name2config
            org_config = data_name2config[config["org_dataset"]]
            task_descriptions = org_config["task_description"], config["task_description"]
            script_path = script_path or org_config["script_path"]
            org2tqas_func = org2tqas_func or org_config["org2tqas_func"]
            post_process_func = post_process_func or org_config["post_process_func"]
        else:
            task_descriptions = config["task_description"], None

        # get functions
        assert org2tqas_func is not None, f"at least one of {dataset} and its original dataset contains `org2tqas_func`"
        org2tqas_func = get_dataset_func(org2tqas_func)
        tf_org_tqas_func = partial(org2tqas_func, task_type=config["task_type"], source=dataset)
        post_process_func = get_dataset_func(post_process_func) if post_process_func else None
        gen_option_func = get_dataset_func(config["gen_option_func"]) if "gen_option_func" in config else None

        pr_sample_train, pr_sample_test = pipeline(tqa_dataset=script_path,
                                                   dataset_info=dataset_info,
                                                    task_descriptions=task_descriptions,
                                                    evidence=config["evidence"],
                                                    shot=shot,
                                                    linearization=linearization,
                                                    tf_org_tqas_func=tf_org_tqas_func, 
                                                    tqas_post_process_func=post_process_func,
                                                    need_gen_option=need_gen_option,
                                                    gen_option_func=gen_option_func,
                                                    option_dump_path=option_dump_path,
                                                    gen_opt_strategies=gen_opt_strategies,
                                                    keep_opt_num=keep_opt_num,
                                                    inverse_option=inverse_option,
                                                    skip_train=skip_train,
                                                    overwrite_cache=overwrite_cache,
                                                    max_data_num=max_data_num,
                                                    max_token_length=max_token_length)
        print(f"saving {dataset}...")
        save_benchmark_data(pr_sample_test, dump_to, config["task_type"], dataset, 
                        shot='one', linearization=linearization, suffix=suffix)
        save_finetune_data(pr_sample_train, dump_to, config["task_type"], dataset, 
                        shot='one', linearization=linearization, suffix=suffix)

if __name__ == '__main__':
    in_argv = get_arguments()
    print(in_argv)
    dump_to_path, raw_json_path, exp_base_path = get_paths(in_argv.machine, keys=["DATABASEPATH", "RAWJSONPATH", "EXPBASEPATH"])
    # initialize the option base path
    option_base_path = os.path.join(exp_base_path, "option_generation/")
    if not os.path.exists(option_base_path):
        os.mkdir(option_base_path)
    data_name2config = parse_dataset_config(in_argv.dataset_config_path, in_argv.base_script_path, raw_json_path)
    max_data_num = in_argv.max_data_num if in_argv.max_data_num != -1 else None
    generate_and_save(datasets=in_argv.datasets, 
                      data_name2config=data_name2config, 
                      dump_to=dump_to_path, 
                      option_base_path=option_base_path,
                      shot=in_argv.shot,
                      linearization=in_argv.linearization, 
                      suffix=in_argv.suffix,
                      gen_opt_strategies=in_argv.gen_opt_strategies,
                      keep_opt_num=in_argv.keep_opt_num,
                      inverse_option=in_argv.inverse_option, 
                      max_data_num=max_data_num,
                      max_token_length=in_argv.max_token_length, 
                      skip_train=in_argv.skip_train,
                      overwrite_cache=in_argv.overwrite_cache)


