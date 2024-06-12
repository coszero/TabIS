
import itertools
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


"""
acc, macro-precision, macro-recall, macro-F1 for multi-class classification 
Task: fact verification
"""
def filter_invalid_samples(pred, gold):
    labels = list(set(gold))
    fil_pred = []
    fil_gold = []
    for p, g in zip(pred, gold):
        if p in labels:
            fil_pred.append(p)
            fil_gold.append(g)
    return fil_pred, fil_gold

def deci2perc(decimal):
    return round(decimal*100, 2)
    
def compute_accuracy(pred, gold):
    assert len(pred) == len(gold) and len(pred) > 0
    all_labels = list(set(gold))
    in_line_hits = [1 if p in all_labels else 0 for p in pred]
    in_line_rate = sum(in_line_hits) / len(pred)

    pred, gold = filter_invalid_samples(pred, gold)
    hits = [1 if p == g else 0 for p, g in zip(pred, gold)]
    acc = sum(hits) / len(hits)
    return {"accuracy": deci2perc(acc), "in_line_rate": deci2perc(in_line_rate)}

def compute_prf1(pred, gold): 
    pred, gold = filter_invalid_samples(pred, gold)
    labels = list(set(gold))
    if len(labels) == 2:
        # if binary classification, must be 0/1
        p, r, f1, _ = precision_recall_fscore_support(gold, pred, pos_label=1, labels=labels, average='binary')
    else:
        p, r, f1, _ = precision_recall_fscore_support(gold, pred, labels=labels, average='macro')
    eval_res = {
        'precision': deci2perc(p), 'recall': deci2perc(r), 'F1': deci2perc(f1)
    }
    return eval_res


"""
tuple-level micro-precision, micro-recall, micro-F1
Task: value description
"""

def intersect_lists(lst1, lst2):
    counter1 = Counter(lst1)
    counter2 = Counter(lst2)
    intersection = counter1 & counter2
    return len(list(intersection.elements()))

def compute_tuple_prf1(preds, golds):
    # Note gold may be duplicate.
    pred_count = 0
    gold_count = 0
    TP_count = 0
    for pred, gold in zip(preds, golds):
        pred_count += len(pred)
        gold_count += len(gold)
        TP_count += intersect_lists(pred, gold)
    precision = TP_count / pred_count if pred_count > 0 else 0
    recall = TP_count / gold_count if gold_count > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision + recall > 0 else 0
    return dict(precision=deci2perc(precision), recall=deci2perc(recall), f1=deci2perc(f1))


"""
hard-accuracy, soft-accuracy of formulas
Task: formula recognition
"""
def calculate_expression(expression):
    # Remove commas from the expression
    expression = expression.replace(',', '')
    expression = expression.replace('%', '/100')
    # Calculate the result
    result = eval(expression)
    return result

def compare_formula(pred, gold, check_cells=False):
    def equal_value_cell(gold_right_eles, pred_right_eles):
        gold_value_cells = [e.replace(',', '') for e in gold_right_eles if any(map(str.isdigit, e))]
        pred_value_cells = [e.replace(',', '') for e in pred_right_eles if any(map(str.isdigit, e))]
        if sorted(gold_value_cells) == sorted(pred_value_cells):
            return 1
        else:
            return 0
    pred_eles = pred.split(" ")
    gold_eles = gold.split(" ")
    if len(pred_eles) <= 3 or pred_eles[1] != '=':
        return 0
    gold_left = calculate_expression(gold_eles[0])
    gold_right = calculate_expression(' '.join(gold_eles[2:]))
    try:
        pred_left = calculate_expression(pred_eles[0])
        pred_right = calculate_expression(' '.join(pred_eles[2:]))
    except:
        return 0
    if abs(gold_left - pred_left) < 0.0001 and abs(gold_right - pred_right) < 0.0001:
        if not check_cells:
            return 1
        else:
            return equal_value_cell(gold_eles[2:], pred_eles[2:])
    else:
        return 0
    
def compute_formula_acc(preds, golds):
    soft_hit_count = 0
    hard_hit_count = 0
    total_count = 0   
    assert len(preds) == len(golds)
    for pred, gold in zip(preds, golds):
        score_soft = compare_formula(pred, gold, check_cells=False)
        score_hard = compare_formula(pred, gold, check_cells=True)
        soft_hit_count += score_soft
        hard_hit_count += score_hard
        total_count += 1
    soft_acc = soft_hit_count / total_count
    hard_acc = hard_hit_count / total_count
    return dict(hard_acc=deci2perc(hard_acc), soft_acc=deci2perc(soft_acc))

def compute_exact_match_score(preds, golds):
    assert len(preds) == len(golds) > 0
    hit_count = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            hit_count += 1
    em_score = hit_count / len(preds)
    return dict(em_score=deci2perc(em_score))

def compute_cell_near_score(pred_res_35, return_details=False):
    # required items in a sample: table_list, tar_pos
    # returned_details: also output the offset count (used for post evaluation)
    vague_hit_count = 0
    hit_count = 0
    offsets2idx = {}
    for idx, samp in enumerate(pred_res_35):
        table_list = samp['sample']['table_list']
        tar_cell_pos = samp['sample']['tar_cell_pos']
        response = samp['response']
        label = samp['label']
        vague_cells = get_vague_cells(table_list, tar_cell_pos)
        if response == label:
            hit_count += 1
        if response in vague_cells:
            vague_hit_count += 1
            poses = vague_cells[response]
            if len(poses) > 1:
                continue
            i, j = poses[0]
            offset_cell = (i+1-tar_cell_pos[0], j+1-tar_cell_pos[1])
            offsets2idx.setdefault(offset_cell, [])
            offsets2idx[offset_cell].append(idx)
    acc = hit_count / len(pred_res_35)
    vague_acc = vague_hit_count / len(pred_res_35)
    metric = dict(em_score=deci2perc(acc), em_o1_score=deci2perc(vague_acc))
    if return_details:
        return metric, offsets2idx
    else:
        return metric

def get_vague_cells(table_list, tar_cell_pos, offset=1):
    # tar_cell_pos: 表格第一行第一列的index是1而不是0
    row_down = tar_cell_pos[0] - offset
    row_down = row_down - 1 if row_down > 0 else 0
    col_down = tar_cell_pos[1] - offset
    col_down = col_down - 1 if col_down > 0 else 0

    row_up = tar_cell_pos[0] + offset + 1
    row_up = row_up - 1 if row_up <= len(table_list) else len(table_list)
    col_up = tar_cell_pos[1] + offset + 1
    col_up = col_up - 1 if col_up <= len(table_list[0]) else len(table_list[0])

    row_range = [*range(row_down, row_up)]
    col_range = [*range(col_down, col_up)]
    vague_cells = {}
    for i, j in itertools.product(row_range, col_range):
        text = table_list[i][j]
        vague_cells.setdefault(text, [])
        vague_cells[text].append((i, j))
    return vague_cells


def compute_rouge_bleu_for_pair(pred, label):
    # also support the rouge score for Chinese
    metric = {}
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    for k, v in result.items():
        metric[k] = round(v["f"] * 100, 4)

    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    metric['bleu-4'] = round(bleu_score * 100, 4)
    return metric

def compute_rouge_bleu(preds, labels):
    agg_metrics = {}
    for pred, label in zip(preds, labels):
        metrics = compute_rouge_bleu_for_pair(pred, label)
        for m, s in metrics.items():
            agg_metrics.setdefault(m, [])
            agg_metrics[m].append(s)
    avg_metrics = {}
    for m, scores in agg_metrics.items():
        avg_metrics[m] = round(sum(scores) / len(scores), 4)
    return avg_metrics

class MetricToolkit():
    name2func = {"acc": compute_accuracy,
                 "prf1": compute_prf1,
                 "tuple-prf1": compute_tuple_prf1,
                 "formula-acc": compute_formula_acc,
                 "exact-match": compute_exact_match_score,
                 "cell-offset-score": compute_cell_near_score,
                 "rouge-bleu": compute_rouge_bleu}
    require_raw_data_funcs = [compute_cell_near_score]
    
    def __init__(self, metrics) -> None:
        self.metric_funcs = [MetricToolkit.name2func[m] for m in metrics]
    
    def compute_metrics(self, model_pred_res):
        
        preds = [r['response'] for r in model_pred_res]
        golds = [r['label'] for r in model_pred_res]
        # remove invalid predictions
        new_preds = []
        new_golds = []
        for pred, gold in zip(preds, golds):
            if pred == "[INVALID]":
                continue
            new_preds.append(pred)
            new_golds.append(gold)
        metrics = {}
        for metric_func in self.metric_funcs:
            if metric_func in MetricToolkit.require_raw_data_funcs:
                metric = metric_func(model_pred_res)
            else:
                metric = metric_func(new_preds, new_golds)
            metrics.update(metric)
        return metrics
    
    def compute_metric_group_by_sample_keys(self, model_pred_res, key):
        # key: must be in the input dict. 
        grouped_pred_res = {}
        for pr in model_pred_res:
            key_ = pr["sample"][key]
            if isinstance(key_, list):
                key_ = '_'.join([str(e) for e in key_])
            grouped_pred_res.setdefault(key_, [])
            grouped_pred_res[key_].append(pr)
        
        grouped_metrics = {}
        for key_, sub_pred_res in grouped_pred_res.items():
            metrics = self.compute_metrics(sub_pred_res)
            grouped_metrics[key_] = metrics
        
        sorted_grouped_metrics = sorted(grouped_metrics.items())
        return sorted_grouped_metrics

