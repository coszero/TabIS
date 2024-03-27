
import os
from typing import List
import copy
import json
import random
from dataclasses import dataclass, field

from data.option.option_prompt import (
    exam_meta_prompt,
    judge_meta_prompt,
    modify_input_prompt,
    modify_answer_prompt,
)
from data.tools import extract_json_content, extract_bracket_content_by_json
from data.data_converter import Tokenizer
from data.option.tools import insert_option
from tab_benchmark.task import eval_ad_hoc_dataset
from tab_benchmark.utils import default_dump_json, get_logger

logger = get_logger(__name__)



def exam_judge(samples: List[dict], dump_to, model_name, url, max_opt_num=5, max_token_length=4000):
    """
    Input instruction-input-output samples and Output samples with candidates and options.

    @param samples: List[dict]. Dict with keys [`instruction`, `input`, `output`]
    @param dump_to: str. The dump path for intermediate outputs.
    @param max_opt_num: int. The maximum number of generated options.
    """

    samples = copy.deepcopy(samples)
    tokenizer = Tokenizer()
    sidx2sample = {}
    for sidx, s in enumerate(samples):    # initialize samples
        s["sidx"] = sidx
        s["candidates"] = []
        s["options"] = []
        sidx2sample[sidx] = s

    if not os.path.exists(dump_to):
        os.makedirs(dump_to)
    
    demo = "Question: Give me a sentence that describes the information of the following cells from the table: 2006 (row 15, column 2); Andriy Sokolovskyy (UKR) (row 15, column 3); 2.36 (row 15, column 4)\nMeta Information:\npage_title: Europa SC High Jump, section_title: Past winners, section_text: Key: Meeting record † = Later disqualified for doping. Antonietta Di Martino was the runner-up with 2.00 m.\nTable:\n| Edition | Year | Men's winner | Mark (m) | Women's winner | Mark (m) |\n| --- | --- | --- | --- | --- | --- |\n| 1st | 1993 | Eugen-Cristian Popescu (ROM) | 2.24 | Šárka Kašpárková (CZE) | 1.95 |\n| 2nd | 1994 | Dalton Grant (GBR) | 2.32 | Iryna Mykhalchenko (UKR) | 1.96 |\n| 3rd | 1995 | Sorin Matei (ROM) | 2.36 | Tatyana Motkova (RUS) | 2.00 |\n| 4th | 1996 | Yuriy Sergiyenko (UKR) | 2.25 | Monika Gollner (AUT) | 1.92 |\n| 5th | 1997 | Patrik Sjöberg (SWE) | 2.28 | Mária Melová-Henkel (SVK) | 1.96 |\n| 6th | 1998 | Artur Partyka (POL) | 2.31 | Yelena Gulyayeva (RUS) | 1.96 |\n| 7th | 1999 | Jan Janků (CZE) | 2.28 | Mária Melová-Henkel (SVK) | 1.95 |\n| 8th | 2000 | Gennadiy Morozov (BLR) | 2.22 | Vita Palamar (UKR) | 1.88 |\n| 9th | 2001 | Sergey Dymchenko (UKR) | 2.31 | Not held | — |\n| — | 2002 | Not held | — | — | — |\n| 10th | 2003 | Stefan Holm (SWE) | 2.34 | Not held | — |\n| 11th | 2004 | Stefan Holm (SWE) | 2.34 | Not held | — |\n| 12th | 2005 | Jaroslav Bába (CZE) | 2.33 | Not held | — |\n| 13th | 2006 | Andriy Sokolovskyy (UKR) | 2.36 | Blanka Vlašić (CRO) | 2.05 |\n| 14th | 2007 | Stefan Holm (SWE) | 2.37 | Venelina Veneva (BUL) † | 2.02 |\n| 15th | 2008 | Stefan Holm (SWE) | 2.34 | Blanka Vlašić (CRO) | 2.04 |\n| 16th | 2009 | Linus Thörnblad (SWE) | 2.36 | Blanka Vlašić (CRO) | 2.00 |\n| 17th | 2010 | Ivan Ukhov (RUS) | 2.38 | Blanka Vlašić (CRO) | 2.04 |\n| 18th | 2011 | Ivan Ukhov (RUS) | 2.38 | Antonietta Di Martino (ITA) | 2.04 |\n| 19th | 2012 | Ivan Ukhov (RUS) | 2.33 | Anna Chicherova (RUS) | 2.00 |\n| 20th | 2013 | Mutaz Essa Barshim (QAT) | 2.36 | Not held | — |\nAnswer: In the men's contest of the 14th Europa SC High Jump, Andrey Sokolovskiy made the record of 2.05 m mark in 2006."
    
    # do exam, gpt-3.5-turbo
    exam_samples = []
    for samp in samples:
        exam_prompt = exam_meta_prompt.format_map(
            {"task_instruct": samp["instruction"],
             "demo": demo,
             "input": samp["input"]
            })
        if tokenizer.compute_tokens(exam_prompt) > max_token_length:
            continue
        exam_sample = dict(prompt=exam_prompt, response="", sidx=samp["sidx"])
        exam_samples.append(exam_sample)
    

    all_pred_res = eval_ad_hoc_dataset(exam_samples, model_name=model_name, url=url, dump_to=dump_to, 
                                       temperature=1, n=max_opt_num, suffix="exam")

    # add candidates to samples, and construct judging prompts
    judge_samples = []
    for pred_res in all_pred_res:
        sidx = pred_res["sample"]["sidx"]
        samp = sidx2sample[sidx]
        candidates_list = list(set([p["content"] for p in pred_res["response"]]))
        candidates_list = sorted(candidates_list, key=lambda x: len(x))
        samp["candidates"] = candidates_list
        candidates = '\n'.join([f"{idx}. "+c for idx, c in enumerate(candidates_list)])
        # candidates = pred_res["response"]
        
        # judge_prompt = judge_meta_prompt.format_map(
        #     {"task_instruct": samp["instruction"], 
        #     "input": samp["input"],
        #     "output": samp["output"],
        #     "candidates": candidates
        #     })
        judge_prompt = judge_meta_prompt.format_map(
            {
                "meta_info": samp["meta_info"],
                "md_table": samp["md_table"],
                # "reference": samp["output"],
                "statements": candidates
            }
        )
        if tokenizer.compute_tokens(judge_prompt) > max_token_length:
            continue

        judge_sample = dict(prompt=judge_prompt, response="", sidx=sidx)
        judge_samples.append(judge_sample)

    # judge, gpt-4
    all_judge_res = eval_ad_hoc_dataset(judge_samples, model_name="gpt-3.5-turbo-16k", dump_to=dump_to,
                                       temperature=0, n=1, suffix="judge")

    # add options to samples
    for judge_res in all_judge_res:
        sidx = judge_res["sample"]["sidx"]
        samp = sidx2sample[sidx]
        candidates_list = samp["candidates"]
        try:
            # parse the json format result
            # the incorrect answers 
            # options = json.loads(extract_json_content(judge_res['response']))["incorrect answers"]
            # options = json.loads(extract_json_content(judge_res['response']))["unfaithful statements"]
            res = json.loads(extract_json_content(judge_res['response']))
            options = [candidates_list[int(i)] for i in res["unfaithful statements"]]
            reasoning = res["reasoning"]
            if len(reasoning) == len(candidates_list):
                # reasoning和option个数对应
                reasoning = [reasoning[int(i)] for i in res["unfaithful statements"]]
            else:
                reasoning = [" | ".join(reasoning) for _ in range(len(options))]
            
        except:
            # print(f"exam-judge parsing failed: {judge_res['response']}")
            options = []
            reasoning = []
        samp["options"] = options
        samp["reasoning"] = reasoning
    
    default_dump_json(samples, os.path.join(dump_to, "samples.json"))
    return samples

def modify_answer_generate(samples: List[dict], dump_to: str, max_opt_num=1, max_token_length=4000):
    samples = copy.deepcopy(samples)
    tokenizer = Tokenizer()
    sidx2sample = {}
    for sidx, s in enumerate(samples):
        s["sidx"] = sidx
        s["candidates"] = []
        s["options"] = []
        sidx2sample[sidx] = s
    
    # modify sampels' input
    modify_list = []
    for sample in samples:
        # prompt = modify_answer_prompt.format_map(
        #     {
        #         "task_instruct": sample["instruction"], 
        #         "input": sample["input"],
        #         "output": sample["output"],
        #     }
        # )
        hcells = ", ".join([p[0] for p in sample["highlighted_cells"]])
        prompt = modify_answer_prompt.format_map(
            {"md_table": sample["md_table"],
            "meta_info": sample["meta_info"],
            "highlighted_cells": hcells,
            "output": sample["output"]}
        )
        if tokenizer.compute_tokens(prompt) > max_token_length:
            continue
        modify_list.append({
            "prompt": prompt,
            "response": sample["output"],
            "sidx": sample['sidx'],
        })
    # if not os.path.exists(dump_to):
    #     os.makedirs(dump_to)
    all_query_res = eval_ad_hoc_dataset(modify_list, model_name="gpt-4", dump_to=dump_to, 
                                       temperature=0, n=1, suffix="mod-answer")

    # concat incorrect options
    for res in all_query_res:
        sidx = res["sample"]["sidx"]
        samp = sidx2sample[sidx]
        try:
            # parse the json format result
            response = json.loads(extract_json_content(res["response"]))
            option = [response["unfaithful statement"]]
            reasoning = [response["reasoning"]]
        except:
            # print(f"mod-answer parsing failed: {res['response']}")
            option = []
            reasoning = []
        samp["options"] = option
        samp["reasoning"] = reasoning

    default_dump_json(samples, os.path.join(dump_to, "samples.json"))
    return samples


def modify_input_generate(samples: List[dict], dump_to: str, max_opt_num=1, max_token_length=4000):
    """ generate incorrect answer by modifying input and performing task.

    Args:
        samples (List[dict]): Dict with keys [`instruction`, `input`, `output`]
        dump_to (str): cache path
        max_opt_num (int, optional): max number of incorrect options to generate. Defaults to 1.
        max_token_length (int, optional): max length of tokens. Defaults to 4000.
    """
    # init samples with None options and idx
    samples = copy.deepcopy(samples)
    tokenizer = Tokenizer()
    sidx2sample = {}
    for sidx, s in enumerate(samples):
        s["sidx"] = sidx
        s["candidates"] = []
        s["options"] = []
        sidx2sample[sidx] = s
    
    # generate prompts
    modify_list = []
    for sample in samples:
        prompt = modify_input_prompt.format_map(
            {
                "task_instruct": sample["instruction"], 
                "input": sample["input"],
                "output": sample["output"],
            }
        )
        if tokenizer.compute_tokens(prompt) > max_token_length:
            continue
        modify_list.append({
            "prompt": prompt,
            "response": sample["output"],
            "sidx": sample['sidx'],
        })
    
    if not os.path.exists(dump_to):
        os.makedirs(dump_to)

    all_query_res = eval_ad_hoc_dataset(modify_list, model_name="gpt-4", dump_to=dump_to, 
                                       temperature=0, n=1, suffix="modify")

    for res in all_query_res:
        sidx = res["sample"]["sidx"]
        samp = sidx2sample[sidx]
        try:
            # parse the json format result
            response = json.loads(extract_json_content(res["response"]))
            option = [response["unfaithful statement"]]
            reasoning = [response["reasoning"]]
        except:
            # print(f"mod-input parsing failed: {res['response']}")
            option = []
            reasoning = []
        samp["options"] = option
        samp["reasoning"] = reasoning

    default_dump_json(samples, os.path.join(dump_to, "samples.json"))
    return samples


@dataclass
class Option():
    text: str
    reasoning: str   # The `reasoning` records how the option is generated.  
    strategy: str

    def __eq__(self, other):
        return self.text == other.text


class OptionGenerator():
    """
    Generate options for instruction-input-output samples.
    - supporting parallel generation via multi-processing.
    """
    STRATEGY2FUNC = {"exam-judge": exam_judge, "mod-input": modify_input_generate, "mod-answer": modify_answer_generate}
    STRATEGY2PROB = {"exam-judge": 0.5, "mod-input": 0.25, "mod-answer": 0.25}
    def __init__(self, strategies: list, keep_opt_num=1) -> None:
        assert all(map(lambda x: x in OptionGenerator.STRATEGY2FUNC, strategies))
        self.strategies = strategies
        self.probs = [OptionGenerator.STRATEGY2PROB[s] for s in self.strategies]
        self.keep_opt_num = keep_opt_num

    def generate(self, iio_samples: List[dict], dump_to, max_token_length=4000):
        
        # step 1: random allocate generation strategies and maximize the diversity TODO.
        strat2sidxs = [[] for _ in range(len(self.strategies))]
        strat_idxs = list(range(len(self.strategies)))
        for sidx in range(len(iio_samples)):
            if self.keep_opt_num > len(self.strategies):
                used_strat_idxs = strat_idxs
            else:
                # for each sample, only use one strategy
                used_strat_idxs = random.choices(strat_idxs, weights=self.probs, k=1)
            for idx in used_strat_idxs:
                strat2sidxs[idx].append(sidx)
        
        # step 2: generate and postprocess options via multiple strategies.
        all_options = [[] for _ in range(len(iio_samples))]
        for strat, sidxs in zip(self.strategies, strat2sidxs):
            gen_option_func = OptionGenerator.STRATEGY2FUNC[strat]
            part_samples = [iio_samples[i] for i in sidxs]
            logger.info(f"generating options via {strat} for {len(part_samples)} samples...")
            this_dump_to = os.path.join(dump_to, f"{strat}/")
            option_samples = gen_option_func(part_samples, dump_to=this_dump_to, 
                                             max_opt_num=3, max_token_length=max_token_length)

            for idx, option_sample in enumerate(option_samples):
                options = option_sample["options"]
                reasonings = option_sample["reasoning"] if "reasoning" in option_sample else [""]*len(options)
                
                new_options = []
                for o, r in zip(options, reasonings):
                    option_text = self.post_process_option_text(str(o))
                    # ignore empty options
                    if option_text:
                        new_options.append(Option(text=option_text, reasoning=r, strategy=strat))
                all_options[sidxs[idx]].extend(new_options)

        # step 3: remove duplicated options
        # TODO: better deduplicate
        all_options = [self.remove_duplicated_options(opts) for opts in all_options]

        for iio_sample, options in zip(iio_samples, all_options):
            if len(options) <= self.keep_opt_num:
                clip_options = options
            else:
                random.shuffle(options)
                clip_options = options[:self.keep_opt_num]

            iio_sample["options"] = [o for o in clip_options]
        
        return iio_samples

    def post_process_option_text(self, text):
        remove_list = ["Output:", "Answer:"]
        for rem in remove_list:
            text = text.replace(rem, "")
        return text.strip()
    
    def remove_duplicated_options(self, options: List['Option']):
        new_opts = []
        for o in options:
            if o not in new_opts:
                new_opts.append(o)
        return new_opts