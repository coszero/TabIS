
import argparse
from tab_benchmark.utils import check_path
from tab_benchmark.task import TableBenchmark, TableBenchmarkConfig

MODELCOLLECTIONPATH = "./config/model_collections.yaml"
LLMSCRIPTPATH = "./tab_benchmark/local_predict.py"

def get_arguments():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str,
                        help="Please specifiy the base path for saving experimental results.")
    parser.add_argument("--eval_models", default=["gpt-3.5-turbo", "gpt-4"], nargs="+",
                        help="Please specifiy the models for evaluations.")
    parser.add_argument("--dataset_path", type=str,
                        help="Specify the base path of datasets for evaluation. Under this directory, one xxx.json represents one dataset to evaluate.")
    parser.add_argument("--max_data_num", default=5000, type=int,
                    help="The max number of test cases for one dataset")
    parser.add_argument("--pool_num", default=10, type=int, 
                    help="The number of CPU pools")
    parser.add_argument("--gpu_devices", default='0', type=str, 
                    help="available gpus")
    parser.add_argument("--max_memory_per_device", default='20', type=int, 
                    help="max memory per GPU device, in GB.")
    parser.add_argument("--eval_group_key", default=None, type=str,
                        help="the key to group the evaluation results")
    parser.add_argument("--seed", default=1, type=int,
                        help="The random seed.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    in_argv = get_arguments()
    check_path(in_argv.exp_path)
    in_argv.model_collection_path = MODELCOLLECTIONPATH
    in_argv.llm_script_path = LLMSCRIPTPATH
    print(in_argv.__dict__)

    task_config = TableBenchmarkConfig(in_argv.__dict__)
    eval_bench_task = TableBenchmark(task_config)
    
    eval_bench_task.eval()