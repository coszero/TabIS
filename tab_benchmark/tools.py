import os
from tab_benchmark.utils import default_load_json

def aggregate_metrics(task_base_path, task_names):
    model2metrics = {}
    for tn in task_names:
        tp = os.path.join(task_base_path, tn, "all_eval_res.json")
        model_metric = default_load_json(tp)
        for model, metric in model_metric:
            model2metrics.setdefault(model, {})
            for m, v in metric.items():
                model2metrics[model].setdefault(m, [])
                model2metrics[model][m].append(v)
    
    model2metric2avg_value = {}
    for model, metrics in model2metrics.items():
        model2metric2avg_value.setdefault(model, {})
        for metric, values in metrics.items():
            avg_value = round(sum(values) / len(values), 2)
            model2metric2avg_value[model][metric] = avg_value

    return model2metric2avg_value

    