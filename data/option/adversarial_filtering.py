import os
import copy
import random
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data.datasets_module import Table
from tab_benchmark.utils import default_load_json, default_dump_md
from tab_benchmark.metric import MetricToolkit


def aggregate_adver_model_res_in_task_dir(task_dir, dataset_name="totto", adver_model_type=["gpt-3.5"]):
    # Given a task directory, aggregate the results of different models on the same sample (including the sample with flipped options)
    # Then compute the score for each sample (the flipped samples are aggregated)
    # Returns: model_detailed_result, statstics
    sfn = [f for f in os.listdir(task_dir) if not f.endswith(".json") and any(map(lambda x: x in f, adver_model_type)) and dataset_name in f]
    index2scores = {}
    index2sample = {}
    model_list = []
    pred_reses = []
    for model_res_fn in sfn:
        model_list.append(model_res_fn.split("_")[-1])
        pred_res = default_load_json(os.path.join(task_dir, model_res_fn, "pred_res.json"))
        pred_reses.append(pred_res)
        for sample_res in pred_res:
            score = int(sample_res["response"] == sample_res["label"])
            index = sample_res["sample"]["index"]
            if index.endswith("-f"):
                index = index[:-2]
            index2scores.setdefault(index, [])
            index2scores[index].append(score)
            if index not in index2sample:
                index2sample[index] = sample_res["sample"]

    index2sum_score = {k: sum(v) for k, v in index2scores.items()}
    # Typically, the number of score is in the range(0, len(model_list)*2+1)
    score_count = [[i, 0] for i in range(len(model_list)*2+1)]
    for score, count in Counter(index2sum_score.values()).items():
        score_count[score] = [score, count]
    sorted_score_count = sorted(score_count, key=lambda x: x[0])
    score = [p[0] for p in sorted_score_count]
    count = [p[1] for p in sorted_score_count]
    acc_count_ratio = [round(sum(count[:idx+1])/sum(count)*100, 2) for idx in range(len(count))]
    count_ratio = [round(count[idx]/sum(count)*100, 2) for idx in range(len(count))]

    model_detailed_result = dict(models=model_list, index2scores=index2scores, index2sum_score=index2sum_score, index2sample=index2sample)
    statstics = dict(count_ratio=count_ratio, acc_count_ratio=acc_count_ratio)

    return model_detailed_result, statstics

def plot_statistics(statstics, key="count_ratio"):
    # plot statistics
    df = pd.DataFrame({
        'score': range(len(statstics[key])),
        key: statstics[key]
    })
    bar = sns.barplot(x='score', y=key, data=df)

    # 在每个条形上显示对应的数值
    for p in bar.patches:
        bar.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()-0.5), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.show()


def conduct_adversarial_filtering(index2sum_score, index2sample, thresholds, sampling_easy=None, dump_hard_for_checking=None):
    """
    Main function of adversarial filtering.
    - Split easy, medium, hard samples according to `thresholds`. `thresholds` must contain three elements: h, m, e,
      such that hard samples are of score [0, h), medium samples of [h, m), and easy samples of [m, e). The samples 
      of which the scores are higher than e will be discarded.
    - Downsample the easy samples if `sampling_easy` is not None.
    - Dump hard samples for manually checking if `dump_hard_for_checking` is not None.
    - Return the split sample indices.
    """

    def classify_sample_by_score(score, thresholds):
        hard_thresh, medium_thresh, easy_thresh = thresholds
        if 0 <= score < hard_thresh:
            return 'hard'
        elif hard_thresh <= score < medium_thresh:
            return 'medium'
        elif medium_thresh <= score < easy_thresh:
            return 'easy'
        else:
            return 'discarded'

    # split samples
    s_type2indices = {}
    for index, sum_score in index2sum_score.items():
        s_type = classify_sample_by_score(sum_score, thresholds)
        s_type2indices.setdefault(s_type, [])
        s_type2indices[s_type].append(index)
    
    # downsampling easy samples
    if sampling_easy is not None:
        assert 0 <= sampling_easy < 1
        new_easy_num = int(len(s_type2indices["easy"])*sampling_easy)
        s_type2indices["easy"] = random.sample(s_type2indices["easy"], new_easy_num)
    
    # dump hard samples for manually checking
    if dump_hard_for_checking is not None:
        hard_samples = [index2sample[idx] for idx in s_type2indices["hard"] if idx in index2sample]
        dump_low_score_samples_for_checking(hard_samples, dump_fp=dump_hard_for_checking)

    print(f"Split result: easy {len(s_type2indices['easy'])}, medium: {len(s_type2indices['medium'])}, hard: {len(s_type2indices['hard'])}")
    return s_type2indices


def recompute_metric_task_dir(task_dir, sub_sample_idxs, metrics, group_key=[], include_flip=False):
    sfn = [f for f in os.listdir(task_dir) if not (f.endswith(".json") or f.endswith(".md"))]
    metric_tool = MetricToolkit(metrics=metrics)
    fn2eval_res = {}
    for fn in sfn:
        model_pred_res = default_load_json(os.path.join(task_dir, fn, "pred_res.json"))
        new_eval_res = recompute_metric_on_sub_samples(model_pred_res, sub_sample_idxs, metric_tool, include_flip, group_key)
        fn2eval_res[fn] = new_eval_res
    return fn2eval_res

def recompute_metric_on_sub_samples(model_pred_res: list, sub_sample_idxs: list[str], metric_tool: MetricToolkit,
                                    include_flip=False, group_key=[]):
    # recompute the model performance on a sub-set of previous predicted samples
    sub_pred_res = []
    for pred_res in model_pred_res:
        index = pred_res["sample"]["index"]
        if index in sub_sample_idxs:
            sub_pred_res.append(pred_res)
        if include_flip and index.endswith('-f') and index[:-2] in sub_sample_idxs:
            sub_pred_res.append(pred_res)

    avg_eval_res = metric_tool.compute_metrics(sub_pred_res)
    all_eval_res = dict(avg=avg_eval_res)
    for key_ in group_key:
        key_eval_res = metric_tool.compute_metric_group_by_sample_keys(sub_pred_res, key=key_)
        all_eval_res[key_] = key_eval_res
    return all_eval_res



def select_low_score_samples(index2sum_score, index2sample, threshold):
    # filtering samples with the relatively low scores 
    to_check_idxs = []
    for index, sum_score in index2sum_score.items():
        if sum_score < threshold:
            to_check_idxs.append(index)
    return [index2sample[idx] for idx in to_check_idxs]


def dump_low_score_samples_for_checking(low_score_samples, dump_fp):

    def highlight_table_cells(grid_table, cell_posits):
        grid_table = copy.deepcopy(grid_table)
        
        for pos in cell_posits:
            
            try:
                # this_cell = grid_table[pos[1]-1][pos[2]-1] or ""
                # grid_table[pos[1]-1][pos[2]-1] = '《' + this_cell + '》'
                this_cell = grid_table[pos[1]][pos[2]] or ""
                grid_table[pos[1]][pos[2]] = '《' + this_cell + '》'
            except:
                print(cell_posits)
                print(len(grid_table), len(grid_table[0]))
                print(grid_table)
                print("-"*50)
                exit()
        return grid_table

    md_files = []
    for idx, sample in enumerate(low_score_samples):
        index = sample["index"]
        md_files.append(f"## sample {idx}-{index}\n")
        md_files.append(f"dataset: {sample['dataset']}\n")
        md_files.append(f"\noption types: {sample['option_types']}\n")

        grid_table = sample["grid_table"]
        if "highlighted_cells" in sample:
            tar_cells = sample["highlighted_cells"]
            # md_files.append(f"{tar_cells}\n\n")
            grid_table = highlight_table_cells(grid_table, tar_cells)

        org_input = sample["input"]
        rem_dem_input = org_input.split("\n\nQuestion: ")[-1]
        text_before_table = rem_dem_input.split("Table:\n")[0].replace("\n", "\n\n")
        # text_after_table = rem_dem_input.split("Options:\n")[-1]
        md_files.append(text_before_table)
        
        md_table = Table.convert_table_data_to_md_str(grid_table)
        md_files.append(md_table)

        md_files.append("\noptions:\n")
        md_files.append(sample["options"].replace("\n", "\n\n"))
        md_files.append(f"\nanswer: {sample['output']}\n")
        md_files.append(f"\nreasoning: {sample['reasoning'][0]}\n")
        md_files.append("="*50)
        md_files.append("**comment**:\n")
        md_files.append("**label**:\n")
        md_files.append("**modification**:\n")
        md_files.append("**opt 1**:\n")
        md_files.append("**opt 2**:\n")
        md_files.append("**true answer**:\n")
    
    default_dump_md(md_files, dump_fp)


