import copy
import os
import shutil

from tab_benchmark.task import eval_ad_hoc_dataset
from typing import List
from datasets_module import Table, TableQASample
from tab_benchmark.utils import get_logger, default_dump_json
from tools import common_generate_meta_summary, common_generate_passage
# from totto.language_repo.language.search_agents.t5.t5_agent_lib import query_to_prompt
from data.option.tools import insert_option, format_list
from data.data_converter import Tokenizer, transform_tqa_dataset_to_zero_shot_iio_dataset
from option.generator import exam_judge, modify_input_generate, OptionGenerator

logger = get_logger(__name__)
exclude_keys = ["table", "question", "answer", "options", "option_types", "caption", "passage"]

"""
org2tqas_funcs [Required]
per each dataset, transform the raw dataset into TableQASample dataset
Args:
    - task_type: str
    - table: Table(table_ins, meta: Optional[dict, str])
    - question: str
    - answer: str
    - source: str
    - passage: Optional[str] = None
    - evidence: Optional[list] = None
"""

def from_tab_percept(sample: dict, task_type, source='') -> 'TableQASample':
    # table perception tasks where the table is regular grid table
    table = Table.from_grid_table(sample["table"])
    other_info = {k: v for k, v in sample.items() if k not in exclude_keys}
    return TableQASample(task_type=task_type, table=table, question=sample["question"], 
                            answer=sample["answer"], options=sample.get("options"), option_types=sample.get("option_types"), 
                            source=source, other_info=other_info)

def from_wikisql(sample: dict, task_type, source='wikisql') -> 'TableQASample':
    question = sample['question']
    answer = sample['answer_text']
    caption = sample['table']['caption']
    section_title = sample['table']['section_title']
    if caption == section_title:
        meta_list = [["caption", caption]]
    else:
        meta_list = [["section title", section_title], ["caption", caption]]
    summary = common_generate_meta_summary(meta_list)
    table = Table.from_header_rows(headers=sample['table']['header'], 
                                    rows=sample['table']['rows'], meta=summary)
    return TableQASample(task_type=task_type, table=table, question=question, 
                        answer=answer, source=source)

def from_wikitq(sample: dict, task_type, source='wikitq') -> 'TableQASample':
    question = sample['question']
    answer = sample['answer_text']
    table = Table.from_header_rows(headers=sample['table']['header'], 
                                    rows=sample['table']['rows'])
    return TableQASample(task_type=task_type, table=table, question=question, 
                        answer=answer, source=source)

def from_fetaqa(sample: dict, task_type, source ='fetaqa') -> 'TableQASample':
    question = sample['question']
    answer = sample['answer_text']
    section_title = sample['section_title']
    page_title = sample['page_title']
    summary = common_generate_meta_summary([["page title", page_title], ["section title", section_title]])
    table = Table.from_header_rows(headers=sample['table']['header'], 
                                    rows=sample['table']['rows'], meta=summary)
    return TableQASample(task_type=task_type, table=table, question=question, 
                        answer=answer, source=source)

def from_wiki(sample: dict, task_type, source='wikisql') -> 'TableQASample':
    # sample loaded from UnifiedSKG scripts: wikisql.py, wikitq.py, hybridqa.py
    assert source in ['wikisql', 'wikitq', 'hybridqa', 'fetaqa']
    question = sample['question']
    if source == 'fetaqa':
        meta = sample['meta']
    elif source == 'hybridqa':
        meta = sample['context']
    else:
        meta = None
    table = Table.from_header_rows(headers=sample['table']['header'], 
                                    rows=sample['table']['rows'], meta=meta)
    answer = sample['answer_text']
    passage = sample['passage'] if source == 'hybridqa' else None
    return TableQASample(task_type=task_type, table=table, question=question, 
                            answer=answer, source=source, passage=passage)

def from_finqa(sample: dict, task_type, source='finqa') -> 'TableQASample':
    # sample loaded from huggingface: dreamerdeo/finqa
    assert source == 'finqa'
    query = sample['question']
    answer = sample['answer']
    table = Table.from_grid_table(sample['table'], table_type='rel')
    evidence = sample['gold_evidence']
    passage = common_generate_passage([["Passage before table", ' '.join(sample['pre_text'])], ["Passage after table", ' '.join(sample['post_text'])]])
    return TableQASample(task_type, table, query, answer, source, passage=passage, evidence=evidence)

def from_hybridqa(sample: dict, task_type, source='hybridqa') -> 'TableQASample':
    question = sample['question']
    answer = sample['answer_text']
    meta = sample['context']
    passage = sample['passage']
    table = Table.from_header_rows(headers=sample['table']['header'], 
                                    rows=sample['table']['rows'], meta=meta)
    return TableQASample(task_type=task_type, table=table, question=question, 
                        answer=answer, source=source, passage=passage)

# def from_totto(sample: dict, task_type, source='totto') -> 'TableQASample':
#     assert source == 'totto'
#     answer = sample['sentence_annotations']['final_sentence'][0]
#     meta = '.'.join([sample['table_page_title'], sample['table_section_title'], sample['table_section_text']])
#     table = Table(totto2table(sample['table']), meta=meta, table_type='rel')
#     # query = str(sample['highlighted_cells'])[1:-1]
    
#     query = str([[table.table_data[x[0]][x[1]].get('content', ''), x[0], x[1]] for x in sample['highlighted_cells']])
#     evidence = None
#     return TableQASample(task_type, table, query, answer, source, evidence=evidence)

def from_totto_c2t(sample: dict, task_type, source="totto_c2t") -> "TableQASample":
    """generate cell-to-text dataset from totto

    Args:
        sample (dict): one data from totto
        task_type (_type_):
        source (str, optional): fixed for this function. for check. Defaults to 'totto'.

    Returns:
        TableQASample: a cell-to-text data
    """
    assert "totto_c2t" in source, (f"get source {source}")
    query_template = "Give me a statement that describes the information of the following cells from the table: {cells}"
    cell_template = "{value} (row {row}, column {col})"
    meta_list = [
        ["page_title", sample['table_page_title']],
        ["section_title", sample["table_section_title"]],
        ["section_text", sample["table_section_text"]],
    ]
    meta_list = [x for x in meta_list if len(x[1]) > 1]
    meta = common_generate_meta_summary(meta_list)
    try:
        table = Table(totto2table(sample["table"]), meta=meta, table_type="rel")
    except:
        return None
    highlighted_cells = [
            [table.table_data[x[0]][x[1]].get("content", ""), x[0]+1, x[1]+1]
            for x in sample["highlighted_cells"]
        ]
    cell_strs = [cell_template.format_map({"value": value, "row": row, "col": col}) for value, row, col in highlighted_cells]
    cells = "; ".join(cell_strs)
    query = query_template.format_map({"cells": cells})
    answer = sample['sentence_annotations']['final_sentence'][0]
    evidence = ["table", "caption"]
    return TableQASample(
        task_type,
        table,
        query,
        answer,
        source,
        evidence=evidence,
        other_info={"index": sample["id"],
                    "highlighted_cells": highlighted_cells},
    )

def from_totto_t2c(sample: dict, task_type, source="totto_t2c") -> "TableQASample":
    """generate cell-to-text dataset from totto

    Args:
        sample (dict): one data from totto
        task_type (_type_):
        source (str, optional): fixed for this function. for check. Defaults to 'totto'.

    Returns:
        TableQASample: a cell-to-text data
    """
    assert "totto_t2c" in source, (f"get source {source}")
    meta_list = [
        ["page_title", sample['table_page_title']],
        ["section_title", sample["table_section_title"]],
        ["section_text", sample["table_section_text"]],
    ]
    meta_list = [x for x in meta_list if len(x[1]) > 1]
    meta = common_generate_meta_summary(meta_list)
    table = Table(totto2table(sample["table"]), meta=meta, table_type="rel")
    query = sample['sentence_annotations']['final_sentence'][0]
    answer = [
        [table.table_data[x[0]][x[1]].get("content", ""), x[0], x[1]]
        for x in sample["highlighted_cells"]
    ]
    
    answer = format_list(answer)
    evidence = ["table", "caption"]
    return TableQASample(
        task_type,
        table,
        query,
        answer,
        source,
        evidence=evidence,
        other_info={"index": sample["id"]},
    )

def from_totto_cp(sample: dict, task_type, source="totto_cp") -> "TableQASample":
    assert "totto_cp" in source, (f"get source {source}")
    meta_list = [
        ["page_title", sample['table_page_title']],
        ["section_title", sample["table_section_title"]],
        ["section_text", sample["table_section_text"]],
    ]
    meta_list = [x for x in meta_list if len(x[1]) > 1]
    meta = common_generate_meta_summary(meta_list)
    table = Table(totto2table(sample["table"]), meta=meta, table_type="rel")
    evidence = ["table"]
    return TableQASample(
        task_type,
        table,
        "summary the caption of the table",
        meta,
        source,
        evidence=evidence,
        other_info={"index": sample["id"]},
    )

def from_hitab_c2t(sample: dict, task_type, source="hitab_c2t") -> "TableQASample":
    """generate cell-to-text dataset from totto

    Args:
        sample (dict): one data from totto
        task_type (_type_):
        source (str, optional): fixed for this function. for check. Defaults to 'totto'.

    Returns:
        TableQASample: a cell-to-text data
    """
    assert "hitab_c2t" in source, (f"get source {source}")
    query_template = "Give me a statement that describes the information of the following cells from the table: {cells}"
    cell_template = "{value} (row {row}, column {col})"
    answer = sample["sub_sentence"]
    meta = sample["table"]["title"]
    table = Table.from_grid_table(sample['table']['texts'], meta=meta, table_type="rel")
    highlighted_cells = get_hitab_highlightcell(sample)
    cell_strs = [cell_template.format_map({"value": value, "row": row+1, "col": col+1}) for value, row, col in highlighted_cells]
    cells = "; ".join(cell_strs)
    query = query_template.format_map({"cells": cells})
    evidence = ["table", "caption"]
    return TableQASample(
        task_type,
        table,
        query,
        answer,
        source,
        evidence=evidence,
        other_info={"index": sample["id"],
                    "highlighted_cells": highlighted_cells},
    )

def from_hitab_t2c(sample: dict, task_type, source="hitab_t2c") -> "TableQASample":
    assert "hitab_t2c" in source, (f"get source {source}")
    answer = get_hitab_highlightcell(sample)
    answer = format_list(answer)
    meta = sample["table"]["title"]
    table = Table(sample['table']['table'], meta=meta, table_type="rel")
    query = sample["sub_sentence"]
    evidence = ["table", "caption"]
    return TableQASample(
        task_type,
        table,
        query,
        answer,
        source,
        evidence=evidence,
        other_info={"index": sample["id"]},
    )

def from_hitab_cp(sample: dict, task_type, source="hitab_cp") -> "TableQASample":
    assert "hitab_cp" in source, (f"get source {source}")
    meta = sample["table"]["title"]
    table = Table(sample['table']['table'], meta=meta, table_type="rel")
    evidence = ["table"]
    return TableQASample(
        task_type,
        table,
        "Summery the table's caption.",
        meta,
        source,
        evidence=evidence,
        other_info={"index": sample["id"]},
    )
"""
post_process_func [Optional]
per each dataset, post process the TableQASample dataset into a new TableQASample dataset
"""

def post_process_t2c(datasets: List['TableQASample'], **kwargs) -> List['TableQASample']:
    for dataset in datasets:
        for x in dataset.options:
            x = format_list(x)
    return datasets

def postprocess_totto(datasets: List['TableQASample'], **kwargs) -> List['TableQASample']:
    return [x for x in datasets if x.bad_table != True]

def postprocess_wikisql(datasets: List['TableQASample'], **kwargs) -> List['TableQASample']:
    # wikisql, wikitq
    # The output is a list. Remove samples of which the output contains multiple elements. (small portion)
    def process_answer(sample):
        if len(sample.answer) > 1:
            sample.answer = None
        else:
            sample.answer = sample.answer[0]
        return sample
    
    new_datasets = []
    for sample in datasets:
        new_sample = process_answer(sample)
        if new_sample.answer is not None:
            new_datasets.append(new_sample)
    return new_datasets


"""
gen_option_func [Optional]
"""        
def gen_option_universal(datasets: List['TableQASample'], **kwargs) -> List['TableQASample']:
    
    if datasets[0].options is None:
        # Note that for PIS tasks, the options are already generated via random sampling. 
        # construct iio dataset
        iio_dataset = transform_tqa_dataset_to_zero_shot_iio_dataset(datasets, 
                                                                    evidence=kwargs["evidence"], 
                                                                    demo_qa=kwargs["demo_qa"],
                                                                    task_description=kwargs["task_description"],
                                                                    linearization=kwargs["linearization"])
        
        dump_to = os.path.join(kwargs["option_dump_path"], f"{kwargs['data_type']}/")
        if kwargs["overwrite_cache"] and os.path.exists(dump_to):
            logger.info(f"overwrite_cache=True, remove files in {dump_to}")
            shutil.rmtree(dump_to)

        # this generates 2*`keep_opt_num`` options and will finally keep `keep_opt_num` options based on the 
        # literal similarity to the golden answer.
        keep_opt_num = kwargs["keep_opt_num"]
        option_generator = OptionGenerator(strategies=kwargs["gen_opt_strategies"], keep_opt_num=2*keep_opt_num)
        new_samples = option_generator.generate(iio_dataset, 
                                dump_to=dump_to,
                                max_token_length=kwargs["max_token_length"])
        
        # add options to TableQASample, the dataset maps one-by-one
        logger.info("using cliping according to length similarity.")
        assert len(iio_dataset) == len(new_samples)
        for tqa_sample, new_sample in zip(datasets, new_samples):
            options = new_sample["options"]
            
            # clip options
            if len(options) > keep_opt_num:
                # select the most similar options in terms of length. 
                sorted_options = sorted(options, key=lambda x: abs(len(x.text) - len(tqa_sample.answer)))
                options = [p for p in sorted_options[:keep_opt_num]]
            tqa_sample.options = [d.text for d in options]
            tqa_sample.option_types = [d.strategy for d in options]
            tqa_sample.other_info["reasoning"] = [d.reasoning for d in options]
        
    # transform to the multi-choice task handle options and answer
    # if using options, need to replace the original answer with option.
    new_dataset = []
    for d in datasets:
        if d.options == [] or d.options is None:
            continue
        # Note that the output answer and options are of type `list`.
        answers, options = insert_option(d.answer, d.options, kwargs["inverse_option"])
        if not kwargs["inverse_option"]:
            assert len(answers) > 0 and len(options) > 0
            d.answer = answers[0]
            d.options = options[0]
            mod_question_based_on_options_totto(d)
            new_dataset.append(d)
        else:
            for idx, (ans, opt) in enumerate(zip(answers, options)):
                new_d = copy.deepcopy(d)
                new_d.answer = ans
                new_d.options = opt
                if idx > 0 and "index" in new_d.other_info:
                    new_d.other_info["index"] += "-f"
                mod_question_based_on_options_totto(new_d)
                new_dataset.append(new_d)
    return new_dataset

def mod_question_based_on_options_totto(totto_sample: TableQASample):
    def split_options(opt_str):
        opt_str = opt_str[3:]
        options = opt_str.split("\nB. ")
        return options
    hc_ques_template = "Based on the following table, which statement about {concat_cell} is accurate?"
    nc_ques_template = "Based on the following table, which statement is accurate?"
    
    option_str = totto_sample.options
    highlighted_cells = [p[0] for p in totto_sample.other_info["highlighted_cells"]]
    options = split_options(option_str)
    opt_1_cells = [c for c in highlighted_cells if c.lower() in options[0].lower()]
    opt_2_cells = [c for c in highlighted_cells if c.lower() in options[1].lower()]
    cells = list(set(opt_1_cells).intersection(opt_2_cells))
    if len(cells) == 0:
        totto_sample.question = nc_ques_template
    else:
        concat_cell = cells[0] if len(cells) == 1 else ", ".join(cells[:-1]) + " and " + cells[-1]
        totto_sample.question = hc_ques_template.format_map({"concat_cell": concat_cell})

def gen_options_totto_c2t(datasets: List['TableQASample'], gen_opt_strategy="", **kwargs) -> List['TableQASample']:
    # add options for fetaqa
    # construct iio dataset
    iio_dataset = transform_tqa_dataset_to_zero_shot_iio_dataset(datasets,
                                                                evidence=kwargs["evidence"],
                                                                task_description=kwargs["task_description"],
                                                                linearization=kwargs["linearization"])
    
    dump_to = os.path.join(kwargs["option_dump_path"], f"{kwargs['data_type']}/")
    if kwargs["overwrite_cache"] and os.path.exists(dump_to):
        logger.info(f"overwrite_cache=True, remove files in {dump_to}")
        shutil.rmtree(dump_to)

    # generate options via modify-input
    new_samples = modify_input_generate(iio_dataset, 
                             dump_to=dump_to,
                             max_opt_num=1,
                             max_token_length=kwargs["max_token_length"])
    
    # add options to TableQASample
    # the dataset maps one-by-one
    assert len(iio_dataset) == len(new_samples) and len(new_samples) == len(datasets), f"length error new_samples{len(new_samples)}, datasets{len(datasets)}"

    new_dataset = []
    for tqa_sample, new_sample in zip(datasets, new_samples):
        if new_sample['options'] == [] or new_sample['options'] is None:
            continue
        tqa_sample.answer, tqa_sample.options = insert_option(tqa_sample.answer, new_sample['options'])
        new_dataset.append(tqa_sample)
    return new_dataset

def gen_options_totto_t2c(datasets: List['TableQASample'], gen_opt_strategy="", **kwargs) -> List['TableQASample']:
    iio_dataset = transform_tqa_dataset_to_zero_shot_iio_dataset(datasets,
                                                                evidence=kwargs["evidence"],
                                                                task_description=kwargs["task_description"],
                                                                linearization=kwargs["linearization"])
    
    dump_to = os.path.join(kwargs["option_dump_path"], f"{kwargs['data_type']}/")
    if kwargs["overwrite_cache"] and os.path.exists(dump_to):
        logger.info(f"overwrite_cache=True, remove files in {dump_to}")
        shutil.rmtree(dump_to)

    # generate options via modify-input
    new_samples = modify_input_generate(iio_dataset,
                             dump_to=dump_to,
                             max_opt_num=1,
                             max_token_length=kwargs["max_token_length"])

    # add options to TableQASample
    # the dataset maps one-by-one
    assert len(iio_dataset) == len(new_samples) and len(new_samples) == len(datasets), \
    f"length error new_samples{len(new_samples)}, datasets{len(datasets)}"

    new_dataset = []
    for tqa_sample, new_sample in zip(datasets, new_samples):
        if new_sample['options'] == [] or new_sample['options'] is None:
            continue
        tqa_sample.answer, tqa_sample.options = insert_option(tqa_sample.answer, new_sample['options'])
        new_dataset.append(tqa_sample)
    return new_dataset

def gen_options_totto_t2c(datasets: List['TableQASample'], gen_opt_strategy="", **kwargs) -> List['TableQASample']:
    iio_dataset = transform_tqa_dataset_to_zero_shot_iio_dataset(datasets,
                                                                evidence=kwargs["evidence"],
                                                                task_description=kwargs["task_description"],
                                                                linearization=kwargs["linearization"])
    
    dump_to = os.path.join(kwargs["option_dump_path"], f"{kwargs['data_type']}/")
    if kwargs["overwrite_cache"] and os.path.exists(dump_to):
        logger.info(f"overwrite_cache=True, remove files in {dump_to}")
        shutil.rmtree(dump_to)

    # generate options via modify-input
    new_samples = modify_input_generate(iio_dataset,
                             dump_to=dump_to,
                             max_opt_num=1,
                             max_token_length=kwargs["max_token_length"])

    # add options to TableQASample
    # the dataset maps one-by-one
    assert len(iio_dataset) == len(new_samples) and len(new_samples) == len(datasets), \
    f"length error new_samples{len(new_samples)}, datasets{len(datasets)}"

    new_dataset = []
    for tqa_sample, new_sample in zip(datasets, new_samples):
        if new_sample['options'] == [] or new_sample['options'] is None:
            continue
        tqa_sample.answer, tqa_sample.options = insert_option(tqa_sample.answer, new_sample['options'])
        new_dataset.append(tqa_sample)
    return new_dataset

def gen_options_hitab_t2c(datasets: List['TableQASample'], gen_opt_strategy="", **kwargs) -> List['TableQASample']:
    # add options for hitab
    # construct iio dataset
    iio_dataset = transform_tqa_dataset_to_zero_shot_iio_dataset(datasets,
                                                                evidence=kwargs["evidence"],
                                                                task_description=kwargs["task_description"],
                                                                linearization=kwargs["linearization"])
    
    dump_to = os.path.join(kwargs["option_dump_path"], f"{kwargs['data_type']}/")
    if kwargs["overwrite_cache"] and os.path.exists(dump_to):
        logger.info(f"overwrite_cache=True, remove files in {dump_to}")
        shutil.rmtree(dump_to)

    # generate options via modify-input
    new_samples = modify_input_generate(iio_dataset, 
                             dump_to=dump_to,
                             max_opt_num=1,
                             max_token_length=kwargs["max_token_length"])
    
    # add options to TableQASample
    # the dataset maps one-by-one
    assert len(iio_dataset) == len(new_samples) and len(new_samples) == len(datasets), f"length error new_samples{len(new_samples)}, datasets{len(datasets)}"

    new_dataset = []
    for tqa_sample, new_sample in zip(datasets, new_samples):
        if new_sample['options'] == [] or new_sample['options'] is None:
            continue
        tqa_sample.answer, tqa_sample.options = insert_option(tqa_sample.answer, new_sample['options'])
        new_dataset.append(tqa_sample)
    return new_dataset

def get_hitab_highlightcell(sample: dict):
    highlightcells = []
    highlightcells += cell_dict2list(sample["linked_cells"]["entity_link"]["top"])
    highlightcells += cell_dict2list(sample["linked_cells"]["entity_link"]["left"])
    highlightcells += cell_dict2list(sample["linked_cells"]["quantity_link"])
    return highlightcells

def cell_dict2list(cell_dict: dict):
    cell_list = []
    for cell in cell_dict.values():
        cell_place = [eval(x) for x in list(cell.keys())]
        cell_content = list(cell.values())
        cell_list += [
            [str(cell_content[i]), cell_place[i][0], cell_place[i][1]]
            for i in range(len(cell_place))
        ]
    return cell_list

# from a gpt reply get options
def get_options(res):
    # select template
    val = eval(res.replace("s' ", r"s\' "))
    if isinstance(val, list) and len(val) == 3:
        return val
    if isinstance(val, dict):
        return list(val.values())[0]
    return

def totto2table(json_data):
    table = [[{'content':cell['value'],
              'rowspan':cell['row_span'], 
              'colspan':cell['column_span'], 
              'is_header': cell['is_header']} for cell in row ] 
              for row in json_data ]
    return table
