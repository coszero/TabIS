import os
import argparse
from config.paths import get_paths
from tab_benchmark.utils import check_path
from tab_benchmark.task import TableBenchmark, TableBenchmarkConfig

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str,
                        help="Please specifiy the tasks for evaluations.")
    parser.add_argument("--mode", default='eval', type=str,
                        help="mode: eval or debug")
    parser.add_argument("--eval_tasks", default=["fact_verification"], nargs="+",
                        help="Please specifiy the tasks for evaluations.")
    parser.add_argument("--eval_models", default=["gpt-3.5-turbo", "gpt-4"], nargs="+",
                        help="Please specifiy the models for evaluations.")
    parser.add_argument("--datasets", default="all", type=str,
                        help="Specify datasets for evaluation."+ \
                            " Use 'all' to work on all datasets or split dataset name with '/'.")
    parser.add_argument("--spec_names", default="all", type=str,
                        help="Specify target experiments for evaluation. " + \
                            "Use 'all' to work on all experiments or split experiment name with a '/'.")
    parser.add_argument("--suffix", type=str, default='',
                        help="the suffix after the saved files. May be used to distinguish different experiments.")
    parser.add_argument("--max_data_num", default=500, type=int,
                    help="The max number of test cases for one dataset")
    parser.add_argument("--pool_num", default=10, type=int, 
                    help="The number of pools")
    parser.add_argument("--gpu_devices", default='0', type=str, 
                    help="available gpus")
    parser.add_argument("--shot", default=["one"], nargs="+",
                        help="Please specify the number of demonstrations in one test case.")
    parser.add_argument("--serialization", default=["markdown"], nargs="+",
                        help="Please specify which serialization you want to evaluate on.")
    parser.add_argument("--eval_group_key", default=None, type=str,
                        help="the key to group the evaluation results")
    parser.add_argument("--seed", default=1, type=int,
                        help="The random seed.")
    parser.add_argument("--machine", default='c10', type=str,
                        help="which machine this script runs on")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    in_argv = get_arguments()
    EXPBASEPATH, DATABASEPATH, DATACOLLECTIONPATH, MODELCOLLECTIONPATH, LLMSCRIPTPATH, LLMMODELPATH = get_paths(in_argv.machine)
    exp_path = os.path.join(EXPBASEPATH, in_argv.task_name)
    check_path(exp_path)
    in_argv.exp_path = exp_path
    in_argv.data_base_path = os.path.join(DATABASEPATH, 'benchmark')
    in_argv.data_collection_path = DATACOLLECTIONPATH
    in_argv.model_collection_path = MODELCOLLECTIONPATH
    in_argv.llm_script_path = LLMSCRIPTPATH
    in_argv.llm_model_path = LLMMODELPATH
    print(in_argv.__dict__)

    task_config = TableBenchmarkConfig(in_argv.__dict__)
    eval_bench_task = TableBenchmark(task_config)
    if in_argv.mode == 'eval':
        eval_bench_task.eval()
    elif in_argv.mode == 'debug':
        eval_bench_task.debug()
    else:
        raise ValueError(f"invalid mode: {in_argv.mode}")