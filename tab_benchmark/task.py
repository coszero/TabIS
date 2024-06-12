import os
import time
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from typing import List

from tab_benchmark.utils import default_dump_json, default_load_json, default_load_yaml, check_path, get_logger
from tab_benchmark.dataset import Dataset
from tab_benchmark.model import model_name2class
from tab_benchmark.metric import MetricToolkit

logger = get_logger(__name__)

class TableBenchmarkConfig():
    base_configs = ['exp_path', 'dataset_path']
    default_config_pairs = []
    def __init__(self, in_dict) -> None:
        for key in TableBenchmarkConfig.base_configs:
            setattr(self, key, in_dict[key])
        for key, value in TableBenchmarkConfig.default_config_pairs:
            setattr(self, key, value)
        for key, value in in_dict.items():
            setattr(self, key, value)
    
    def dump_to(self, dir_path, file_name='benchmark_setting.json'):
        dump_fp = os.path.join(dir_path, file_name)
        default_dump_json(self.__dict__, dump_fp)


class TableBenchmark():
    def __init__(self, task_config: TableBenchmarkConfig):
        self.config = task_config
        self.config.dump_to(self.config.exp_path)
        # load data
        self.load_eval_data()
        # load models
        self.load_eval_model()

    def load_eval_data(self):
        # load evaluation data
        data_fns = [fn for fn in os.listdir(self.config.dataset_path) if fn.endswith(".json")]
        logger.info(f"valid datasets: {data_fns}")

        data_collections = []
        for data_fn in data_fns: 
            spec_name = data_fn[:-5]
            url = os.path.join(self.config.dataset_path, data_fn)
            dataset = Dataset(spec_name=spec_name, format="prompt-response", verbalizer="table-qa-mc", 
                              metric=["exact-match"], url=url, max_data_num=self.config.max_data_num)
            data_collections.append(dataset)
        self.eval_dataset = data_collections

    def load_eval_model(self):
        model_collections = default_load_yaml(self.config.model_collection_path)
        models = []
        for m in self.config.eval_models:
            if m not in model_collections:
                raise NotImplementedError(f"{m} is not a valid model.")
            model_config = model_collections[m]
            model = model_name2class[model_config["class_name"]].from_setting(spec_name=m, setting=model_config["setting"])
            models.append(model)
        self.models = models

    def eval(self):
        all_eval_res = []
        for dataset in self.eval_dataset:
            for model in self.models:
                logger.info(f"evaluate model={model.spec_name} on dataset={dataset.spec_name}")
                eval_task = EvalTask(dataset=dataset, model=model)
                eval_res = eval_task.eval_model_on_dataset(
                    dump_base_path=self.config.exp_path, 
                    max_memory_per_device=self.config.max_memory_per_device,
                    gpu_devices=self.config.gpu_devices,
                    pool_num=self.config.pool_num, 
                    script_path=self.config.llm_script_path, 
                    eval_group_key=self.config.eval_group_key)
                all_eval_res.append(eval_res)
        all_eval_path = os.path.join(self.config.exp_path, 'all_eval_res.json')
        self.dump_inherit_old_eval_results(all_eval_res, all_eval_path)
        logger.info(all_eval_res)

    def dump_inherit_old_eval_results(self, all_eval_res, all_eval_path):
        # If old eval res exists in the directory, inherit results on old models.
        if os.path.exists(all_eval_path):
            old_eval_res = default_load_json(all_eval_path)
        else:
            old_eval_res = []

        old_eval_task_names = [r[0] for r in old_eval_res]
        for eval_res in all_eval_res:
            if eval_res[0] not in old_eval_task_names:
                old_eval_res.append(eval_res)
        
        default_dump_json(old_eval_res, all_eval_path)


class EvalTask():
    def __init__(self, dataset: Dataset, model) -> None:
        self._dataset = dataset
        self.model = model
        self.evaluator = MetricToolkit(self._dataset.metric)
        self.eval_task_name = f"{self._dataset.spec_name}_{self.model.spec_name}"

    def eval_model_on_dataset(self, dump_base_path, max_memory_per_device=10, skip_eval=False, gpu_devices=None, pool_num=1, script_path=None, eval_group_key=None):
        """
        There are two kinds of models currently: 
            1. proprietary models (GPT-3.5-turbo, e.g.). The models are only available via API calls. 
            2. local models (ChatGLM, LLaMa, e.g.). The models run locally.
        """
        eval_dump_base_path = f"{dump_base_path}/{self.eval_task_name}/"
        
        if self.model.__class__.NAME in ['openllm']:
            all_reply_info = self.get_model_output_gpu(
                verbalizer=self._dataset.verbalizer, 
                gpu_devices=gpu_devices, 
                script_path=script_path, 
                max_memory_per_device=max_memory_per_device,
                run_dump_path=eval_dump_base_path)
        else:
            all_reply_info = self.get_model_output_cpu(
                verbalizer=self._dataset.verbalizer, 
                pool_num=pool_num, 
                run_dump_path=eval_dump_base_path)

        if skip_eval:
            return all_reply_info
        eval_res = self.evaluator.compute_metrics(all_reply_info)
        if eval_group_key:
            grouped_eval_res = self.evaluator.compute_metric_group_by_sample_keys(all_reply_info, eval_group_key)
        else:
            grouped_eval_res = {}
        task_name_eval_res = [self.eval_task_name, eval_res, grouped_eval_res]
        eval_res_dump_path = os.path.join(eval_dump_base_path, 'eval_res.json')
        default_dump_json(task_name_eval_res, eval_res_dump_path)
        return task_name_eval_res
    
    def get_model_output_gpu(self, verbalizer, gpu_devices, script_path, max_memory_per_device, run_dump_path):
        check_path(run_dump_path)
        all_reply_info_path = os.path.join(run_dump_path, 'pred_res.json')
        if os.path.exists(all_reply_info_path):
            all_reply_info = default_load_json(all_reply_info_path)
        else:
            all_reply_info = self.model.forward_all_samples(self._dataset.dataset, gpu_devices, script_path, max_memory_per_device,
                                                        verbalizer=verbalizer, dump_dir=run_dump_path)
            default_dump_json(all_reply_info, all_reply_info_path)
        return all_reply_info

    def get_model_output_cpu(self, verbalizer, pool_num, run_dump_path):
        check_path(run_dump_path)
        res_dump_path = os.path.join(run_dump_path, 'pred_res.json')
        if pool_num == 1:
            if os.path.exists(res_dump_path):
                all_reply_info = default_load_json(res_dump_path)
                done_num = len(all_reply_info)
                logger.info(f"start from: index {done_num}")
            else:
                all_reply_info = []
                done_num = 0
                logger.info(f"start from scratch")
            
            for idx, sample in enumerate(tqdm(self._dataset.dataset[done_num:])):
                data_reply_info = self.model.forward(sample, verbalizer=verbalizer)
                all_reply_info.append(data_reply_info)
                if idx % 20 == 0:
                    default_dump_json(all_reply_info, res_dump_path)
        else:
            start_time = time.time()
            data_pairs = [[idx, sample] for idx, sample in enumerate(self._dataset.dataset)]
            output_path = os.path.join(run_dump_path, 'output/')
            if os.path.exists(output_path):
                sorted_fn = sorted(os.listdir(output_path), key=lambda x: int(x.split('.')[0]))
                indices = [int(x.split('.')[0]) for x in sorted_fn]
                data_pairs = [data_pairs[i] for i in range(len(data_pairs)) if i not in indices]
                logger.info(f"working on {len(data_pairs)} remaining samples under {output_path}.")
            check_path(output_path)
            
            from tqdm.contrib.concurrent import process_map
            func = partial(self.model.forward, verbalizer=verbalizer, with_indices=True, dump_dir=output_path)
            process_map(func, data_pairs, max_workers=pool_num, chunksize=1)
            sorted_fn = sorted(os.listdir(output_path), key=lambda x: int(x.split('.')[0]))
            all_reply_info = []
            for fn in sorted_fn:
                fp = os.path.join(output_path, fn)
                reply_info = default_load_json(fp)
                all_reply_info.append(reply_info)
        default_dump_json(all_reply_info, res_dump_path)
        return all_reply_info
    

def eval_ad_hoc_dataset(samples: List[dict], model_name: str, dump_to: str, url=None, temperature=0, n=1, metric='', suffix=''):
    """
    - Evaluate ad_hoc samples with multi-processing tools.
    - Must specify a dump path to save intermediate outputs.
    - One sample must have key `prompt` and `response`.
    - Model_name must be in GPT families.
    """
    dataset = Dataset.from_ad_hoc_dataset(samples, shuffle=False, metric=metric, max_data_num=None, suffix=suffix)
    model = model_name2class["APIAgent"](model_name, model_name, url=url, temperature=temperature, n=n)
    eval_task = EvalTask(dataset=dataset, model=model)
    if metric == '':
        skip_eval = True
    else:
        skip_eval = False
    all_reply_info = eval_task.eval_model_on_dataset(dump_base_path=dump_to, skip_eval=skip_eval, pool_num=10)
    return all_reply_info
