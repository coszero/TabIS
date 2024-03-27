import random
import copy
from collections import Counter
from typing import List
from tab_benchmark.utils import default_load_txt

def insert_option(golden: str, wrong_options, inverse_option=False):
    # random insert the golden answer to options
    assert isinstance(wrong_options, list)
    
    opt_num = len(wrong_options)
    index = random.randint(0, opt_num)  # the index to insert golden answer.
    if inverse_option and (opt_num-index != index):
        # if inverse the option, the mirrored index will be provided
        indices = [index, opt_num-index]
    else:
        indices = [index]

    answers = []
    options = []
    for index in indices:
        options_cp = copy.deepcopy(wrong_options)
        options_cp.insert(index, golden)
        answer = chr(index + 65) 
        options_cp = [f"{chr(x+65)}. {options_cp[x]}" for x in range(opt_num+1)]
        options_str = "\n".join(options_cp)
        answers.append(answer)
        options.append(options_str)

    return answers, options

def format_list(unformated):
    """make a list to string. Make sure strings in "" not ''.

    Args:
        unformated_list (list): source list

    Returns:
        str: Formated list in string formate
    """
    if isinstance(unformated, str):
        return unformated
    formated_list = []
    for item in unformated:
        if isinstance(item, list):
            formated_list.append(format_list(item))
        elif isinstance(item, str) or item == None:
            formated_list.append(f'"{item}"')
        else:
            formated_list.append(str(item))
    return '[' + ', '.join(formated_list) + ']'

def compute_check_option_acc(md_path):
    md_files = default_load_txt(md_path)

    scores = []
    for row in md_files:
        if row.startswith("Score:"):
            score = row[6:]
            score = score.replace("\n").strip()
            try:
                score = int(score)
                scores.append(score)
            except:
                continue
    
    

