"""
tabular perception 
Note that all tabular perception tasks are generated via self supervision.
Generate raw datasets (json format) for tasks in tabular perception.
Input: table corpus
Output: generated raw samples for each task. a sample is a dict with following 
        keys: ['table', 'question', 'answer']
""" 

import copy
import random
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Dict
from itertools import product
from collections import Counter
from data.tools import train_test_split_sample
from data.datasets_module import Table

def get_random_offset_cell_positions(table, x, y, num, x_range, y_range, exclude_cells=[]):
    # sample `num` offset positions.
    new_cells = []
    new_poses = []
    org_cell = table[x][y]
    trial = 0
    while trial <= 100:
        m = int(np.random.normal(0, 1))
        n = int(np.random.normal(0, 1))
        if x_range[0]<= m+x <x_range[1] and y_range[0]<= y+n <y_range[1] \
            and table[m+x][n+y] != org_cell and table[m+x][n+y] not in new_cells \
            and table[m+x][n+y] not in exclude_cells: 
            new_cells.append(table[m+x][n+y])
            new_poses.append((m+x, n+y))
        trial += 1
    return new_cells[:num], new_poses[:num] 

def get_random_offset_rows_cols(x, num, x_range):
    # sample `num` offset positions. Don't need to deduplicate.
    new_poses = []
    trial = 0
    while trial <= 100:
        m = int(np.random.normal(0, 1))
        if m !=0 and x_range[0]<= m+x <x_range[1] and m+x not in new_poses: 
            new_poses.append(m+x)
        if len(new_poses) >= num:
            break
        trial += 1
    return new_poses[:num]

def random_mix_two_lists(anchor_cell, anchor_list, tar_lists, keep_anchor_ratio=0.5, mix_ratio=0.5):
    def keep_anchor(ratio):
        p = np.random.uniform(0, 1)
        if p < ratio:
            return True
        else:
            return False
    
    new_tar_lists = []
    for l in tar_lists:
        new_tar_list = []
        for e_a, e_t in zip(anchor_list, l):
            if e_a == anchor_cell:
                new_tar_list.append(e_a)
            else:
                new_ele = e_a if keep_anchor(keep_anchor_ratio) else e_t
                new_tar_list.append(new_ele)
        if keep_anchor(mix_ratio):
            new_tar_lists.append(new_tar_list)
        else:
            new_tar_lists.append(l)
    
    return new_tar_lists
    

def filter_samples(samples):
    return [s for s in samples if len(s["options"]) > 0]

def construct_samples_table_size_detection(tables: List[List[List[str]]], test_size=0.2, seed=0) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    question_template = "What is the count of rows and columns in the table?"
    answer_template = "{row_num} rows, {col_num} columns"
    samples = []
    for table in tables:
        gold_row_num = len(table)
        gold_col_num = len(table[0])
        question = question_template
        answer = answer_template.format_map({"row_num": gold_row_num, "col_num": gold_col_num})
        samples.append(dict(table=table, question=question, answer=answer))
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_positional_cell_lookup(tables: List[List[List[str]]], test_size=0.2, seed=0, 
                                            is_mc=False, option_num=3, is_sample=True) -> List[dict]:
    # if is_sample=False, all possible samples will be generated.
    random.seed(seed)
    np.random.seed(seed)
    # random.shuffle(tables)
    question_template = "What is the content of the cell located at row {row_num} and column {col_num}?"
    # question_template = "请输出第{row_num}行第{col_num}列的单元格的内容。"
    answer_template = "{cell_content}"
    samples = []
    for table in tqdm(tables):
        row_num, col_num = len(table), len(table[0])
        if is_sample:
            row = random.sample(range(row_num), 1)[0]
            col = random.sample(range(col_num), 1)[0]
            iter_pair = [[row, col]]
        else:
            iter_pair = product(range(row_num), range(col_num))
        for row, col in iter_pair:
            content = table[row][col]
            question = question_template.format_map({"row_num": row+1, "col_num": col+1})
            answer = answer_template.format_map({"cell_content": content})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                new_cells, new_poses = get_random_offset_cell_positions(table, row, col, option_num, [0, row_num], [0, col_num])
                if len(new_cells) == 0:
                    continue
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=new_cells, option_types=option_types, opt_cell_pos=new_poses, evidence_pos=[row, col])
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_positional_cell_lookup_for_exp(tables: List[List[List[str]]], seed=0) -> List[dict]:
    # random.seed(seed)
    # random.shuffle(tables)
    question_template = "What is the content of the cell located at row {row_num} and column {col_num}?"
    # question_template = "请输出第{row_num}行第{col_num}列的单元格的内容。"
    answer_template = "{cell_content}"
    samples = []
    for table in tables:
        row_num, col_num = len(table), len(table[0])
        for row, col in product(range(row_num), range(col_num)):
            content = table[row][col]
            question = question_template.format_map({"row_num": row+1, "col_num": col+1})
            answer = answer_template.format_map({"cell_content": content})
            tar_cell_pos = [row+1, col+1]
            table_list = table
            samples.append(dict(table=table, question=question, answer=answer,
                                tar_cell_pos=tar_cell_pos, table_list=table_list))
    # samples[0]["answer"] = "亚运会"
    return dict(train=[samples[18]], validation=samples[row_num*col_num:])

def construct_sample_relative_cell_lookup(tables: List[List[str]], test_size=0.2, seed=0, is_mc=False, option_num=3, is_sample=False) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    # Note that there are multiple question templates in this task.
    # Each question has a corresponding gold answer template.
    prefix_question = "The anchor cell is '{anchor_cell}' in row {row} and column {col}. "
    question_template = [
        "What is the content of the first cell below the anchor cell within the same column？",
        "What is the content of the first cell above the anchor cell within the same column？",
        "What is the content of the first cell left to the anchor cell within the same row？",
        "What is the content of the first cell right to the anchor cell within the same row？"
        ]
    # the gold anwser refers to the row/col offset of the target cell
    gold_answer = [[1, 0], [-1, 0], [0, -1], [0, 1]]
    answer_template = "{cell_content}"

    one_qtype_limit = int(len(tables)/len(question_template))

    samples = []
    for idx, table in enumerate(tqdm(tables)):
        # sample an anchor cell
        row_num = len(table)
        col_num = len(table[0])
        if is_sample:
            row = random.sample(range(1, row_num-1), 1)[0]
            col = random.sample(range(1, col_num-1), 1)[0]
            # choose a question template
            qt_idx = idx // one_qtype_limit
            iter_pair = [[row, col, qt_idx]]
        else:
            iter_pair = product(range(row_num), range(col_num), range(len(question_template)))
        for row, col, qt_idx in iter_pair:
            anchor_cell = table[row][col]
            qt_idx = qt_idx if qt_idx < len(question_template) else len(question_template) - 1
            sam_question_template = prefix_question+ question_template[qt_idx]
            sam_gold_answer = gold_answer[qt_idx]
            ans_row, ans_col = row+sam_gold_answer[0], col+sam_gold_answer[1]
            try:
                # For is_sampling=False, the position of ans_cell may be out of range on the edge.
                assert ans_row >= 0 and ans_col >= 0
                cell_content = table[ans_row][ans_col]
            except:
                continue
            question = sam_question_template.format_map({"anchor_cell": anchor_cell, "row": row+1, "col": col+1})
            answer = answer_template.format_map({"cell_content": cell_content})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                # Note that the options should not be the same as the anchor cell and the target cell.
                new_cells, new_poses = get_random_offset_cell_positions(table, ans_row, ans_col, option_num, [0, row_num], [0, col_num], exclude_cells=[anchor_cell])
                if len(new_cells) == 0:
                    continue
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=new_cells, option_types=option_types, opt_cell_pos=new_poses,  evidence_pos=[ans_row, ans_col], qt_idx=qt_idx)
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_relative_cell_lookup_for_exp(tables: List[List[str]], train_sample_idx: List[int], seed=0) -> List[dict]:
    random.seed(seed)
    random.shuffle(tables)
    # Note that there are multiple question templates in this task.
    # Each question has a corresponding gold answer template.
    def get_valid_gold_answer_idxs(row, col, row_num, col_num, gold_answer):
        valid_gold_answer_idxs = []
        for idx, (row_offset, col_offset) in enumerate(gold_answer):
            if row + row_offset < 0 or row + row_offset >= row_num:
                continue
            if col + col_offset < 0 or col + col_offset >= col_num:
                continue
            valid_gold_answer_idxs.append(idx)
        return valid_gold_answer_idxs
    
    prefix_question = "The anchor cell is '{anchor_cell}' in row {row} and column {col}. "
    question_template = [
        "What is the content of the first cell below the anchor cell within the same column？",
        "What is the content of the first cell above the anchor cell within the same column？",
        "What is the content of the first cell left to the anchor cell within the same row？",
        "What is the content of the first cell right to the anchor cell within the same row？"
        ]
    # the gold anwser refers to the row/col offset of the target cell
    gold_answer = [[1, 0], [-1, 0], [0, -1], [0, 1]]
    answer_template = "{cell_content}"

    samples = []
    train_sample = None
    # random.shuffle(tables)
    for tidx, table in enumerate(tables):
        # sample an anchor cell
        row_num = len(table)
        col_num = len(table[0])

        for row, col in product(range(row_num), range(col_num)):
            anchor_cell = table[row][col]
            valid_gold_answer_idxs = get_valid_gold_answer_idxs(row, col, row_num, col_num, gold_answer)

            # print(row, col, len(valid_gold_answer_idxs))
            for qt_idx in valid_gold_answer_idxs:
                sam_question_template = prefix_question+ question_template[qt_idx]
                sam_gold_answer =  gold_answer[qt_idx]
                cell_content = table[row+sam_gold_answer[0]][col+sam_gold_answer[1]]
                question = sam_question_template.format_map({"anchor_cell": anchor_cell, "row": row+1, "col": col+1})
                answer = answer_template.format_map({"cell_content": cell_content})
                sample = dict(table=table, question=question, answer=answer,
                                    table_list=table, tar_pos=sam_gold_answer, anchor_pos=[row, col])
                samples.append(sample)
                if [tidx, row, col, qt_idx] == train_sample_idx:
                    train_sample = sample
    return dict(train=[train_sample], validation=samples[168:])
    
def construct_sample_positional_row_lookup(tables: List[List[str]], test_size=0.2, seed=0, is_mc=False, option_num=3, is_sample=True) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    
    question_template = "What are the contents of the cells in row {row}? In the options, cells are delineated by '|'."
    answer_template = "{cell_contents}"

    samples = []
    for table in tables:
        row_num = len(table)
        if is_sample:
            row = random.sample(range(row_num), 1)[0]
            iter_pair = [row]
        else:
            iter_pair = range(row_num)
        for row in iter_pair:
            cell_contents = "|".join(table[row])
            question = question_template.format_map({"row": row+1})
            answer = answer_template.format_map({"cell_contents": cell_contents})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                new_poses = get_random_offset_rows_cols(row, option_num, [0, row_num])
                options = ["|".join(table[r]) for r in new_poses]
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=options, option_types=option_types, opt_row_pos=new_poses,  evidence_pos=row)
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_positional_column_lookup(tables: List[List[str]], test_size=0.2, seed=0, is_mc=False, option_num=3, is_sample=True) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)

    question_template = "What are the contents of the cells in column {col}? In the options, cells are delineated by '|'."
    answer_template = "{cell_contents}"

    samples = []
    for table in tables:
        col_num = len(table[0])
        if is_sample:
            col = random.sample(range(col_num), 1)[0]
            iter_pair = [col]
        else:
            iter_pair = range(col_num)
        for col in iter_pair:
            cell_contents = "|".join([r[col] for r in table])
            question = question_template.format_map({"col": col+1})
            answer = answer_template.format_map({"cell_contents": cell_contents})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                new_poses = get_random_offset_rows_cols(col, option_num, [0, col_num])
                options = ["|".join([r[c] for r in table]) for c in new_poses]
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=options, option_types=option_types, opt_col_pos=new_poses, evidence_pos=col)
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_relative_row_lookup(tables: List[List[str]], test_size=0.2, seed=0, is_mc=False, option_num=3, is_sample=True) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    prefix_question = "The anchor cell is '{anchor_cell}' in row {row} and column {col}. "
    question_template = [
        "What are the contents of the cells within the same row as the anchor cell? ",
        "What are the contents of the first row above the anchor cell? ",
        "What are the contents of the first row below the anchor cell? "
    ]
    suffix_question = " In the options, cells are delineated by '|'."
    gold_answer = [0, -1, 1]
    answer_template = "{cell_contents}"

    one_qtype_limit = int(len(tables)/len(question_template))

    samples = []
    for idx, table in enumerate(tables):
        row_num, col_num = len(table), len(table[0])
        if is_sample:
            row = random.sample(range(1, row_num-1), 1)[0]
            col = random.sample(range(col_num), 1)[0]
            qt_idx = idx // one_qtype_limit
            iter_pair = [[row, col, qt_idx]]
        else:
            iter_pair = product(range(row_num), range(col_num), range(len(question_template)))
        for row, col, qt_idx in iter_pair:
            qt_idx = qt_idx if qt_idx < len(question_template) else len(question_template) - 1
            anchor_cell = table[row][col]
            sam_question_template = prefix_question + question_template[qt_idx] + suffix_question
            ans_row = gold_answer[qt_idx] + row
            try:
                assert ans_row >= 0
                cell_contents = "|".join(table[ans_row])
            except:
                continue
            question = sam_question_template.format_map({"anchor_cell": anchor_cell, "row": row+1, "col": col+1})
            answer = answer_template.format_map({"cell_contents": cell_contents})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                new_poses = get_random_offset_rows_cols(ans_row, option_num, [0, row_num])
                if len(new_poses) == 0:
                    continue
                tar_rows = [table[r] for r in new_poses]
                new_tar_rows = random_mix_two_lists(anchor_cell, table[ans_row], tar_rows, keep_anchor_ratio=0.5, mix_ratio=0)
                concat_new_tar_rows = ["|".join(r) for r in new_tar_rows]
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=concat_new_tar_rows, option_types=option_types, opt_row_pos=new_poses,  evidence_pos=ans_row)
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_relative_column_lookup(tables: List[List[str]], test_size=0.2, seed=0, is_mc=False, option_num=3, is_sample=True) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    prefix_question = "The anchor cell is '{anchor_cell}' in row {row} and column {col}. "
    question_template = [
        "What are the contents of the cells within the same column as the anchor cell?",
        "What are the contents of the first column left to the anchor cell?",
        "What are the contents of the first column right to the anchor cell?"
    ]
    suffix_question = " In the options, cells are delineated by '|'."
    gold_answer = [0, -1, 1]
    answer_template = "{cell_contents}"

    one_qtype_limit = int(len(tables)/len(question_template))

    samples = []
    for idx, table in enumerate(tables):
        row_num, col_num = len(table), len(table[0])
        if is_sample:
            row = random.sample(range(row_num), 1)[0]
            col = random.sample(range(1, col_num-1), 1)[0]
            # choose a question template
            qt_idx = idx // one_qtype_limit
            iter_pair = [[row, col, qt_idx]]
        else:
            iter_pair = product(range(row_num), range(col_num), range(len(question_template)))
        for row, col, qt_idx in iter_pair:
            qt_idx = qt_idx if qt_idx < len(question_template) else len(question_template) - 1
            anchor_cell = table[row][col]
            sam_question_template = prefix_question + question_template[qt_idx] + suffix_question
            ans_col = gold_answer[qt_idx] + col
            try:
                assert ans_col >= 0
                cell_contents = "|".join([r[ans_col] for r in table])
            except:
                continue
            question = sam_question_template.format_map({"anchor_cell": anchor_cell, "row": row+1, "col": col+1})
            answer = answer_template.format_map({"cell_contents": cell_contents})
            if not is_mc:
                new_sample = dict(table=table, question=question, answer=answer)
            else:
                new_poses = get_random_offset_rows_cols(ans_col, option_num, [0, col_num])
                if len(new_poses) == 0:
                    continue
                tar_cols = [[r[c] for r in table] for c in new_poses]
                new_tar_cols = random_mix_two_lists(anchor_cell, [r[ans_col] for r in table], tar_cols, keep_anchor_ratio=0.5, mix_ratio=0)
                concat_new_tar_cols = ["|".join(r) for r in new_tar_cols]
                option_types = ["random-sample"]*option_num
                new_sample = dict(table=table, question=question, answer=answer, 
                                options=concat_new_tar_cols, option_types=option_types, opt_row_pos=new_poses,  evidence_pos=ans_col)
            samples.append(new_sample)
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_empty_cell_location(tables: List[List[str]], test_size=0.2, seed=0) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    # only applied to tables without empty cells.
    def check_empty_cell(table):
        return any(map(lambda x: x == '', [c for r in table for c in r]))
    question_template = "What is the location of the empty cell that contains empty value?"
    answer_template = "row {row}, column {column}"

    with_missing_count = 0
    samples = []
    tables = copy.deepcopy(tables)
    for table in tables:
        if check_empty_cell(table):
            with_missing_count += 1
            continue
        row_num, col_num = len(table), len(table[0])
        row = random.sample(range(row_num), 1)[0]
        col = random.sample(range(col_num), 1)[0]
        table[row][col] = ''

        answer = answer_template.format_map({"row": row, "column": col})
        samples.append(dict(table=table, question=question_template, answer=answer))
    print(f"There are {with_missing_count} (out of {len(tables)}) tables that have empty cells.")
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)

def construct_sample_column_name_identification(tables: List[List[str]], test_size=0.2, horizontal=True, seed=0) -> List[dict]:
    random.seed(seed)
    # random.shuffle(tables)
    # horizontal=True，only applied for herizontal tables
    # when horizontal=False, the table will be transposed and the problem would be row name identification

    def detect_once_values(table, horizontal=True):
        # if horizontal, not the first row
        # if vertical, not the first column
        cell2posits = {}
        for row_idx, row in enumerate(table):
            if horizontal and row_idx == 0:
                continue
            for col_idx, cell in enumerate(row):
                if not horizontal and col_idx == 0:
                    continue
                cell2posits.setdefault(cell, [])
                cell2posits[cell].append([row_idx, col_idx])

        return [[p[0], p[1][0]] for p in cell2posits.items() if len(p[1]) == 1]
    
    question_template_h = "What is the column name of the value {cell_content}?"
    answer_template_h = "{column_name}"
    question_template_r = "What is the row name of the value {cell_content}?"
    answer_template_r = "{row_name}"

    no_once_value_count = 0
    samples = []
    for table in tables:
        if not horizontal:
            table = Table.transpose_grid_table(table)
        once_values = detect_once_values(table, horizontal=horizontal)
        cell_content, position = random.sample(once_values, 1)[0]
        
        if horizontal:
            column_name = table[0][position[1]]
            question = question_template_h.format_map({"cell_content": cell_content})
            answer = answer_template_h.format_map({"column_name": column_name})
        else:
            row_name = table[position[0]][0]
            question = question_template_r.format_map({"cell_content": cell_content})
            answer = answer_template_r.format_map({"row_name": row_name})

        samples.append(dict(table=table, question=question, answer=answer))

    print(f"There are {no_once_value_count} (out of {len(tables)}) tables that have no once-values.")
    train_samples, test_samples = train_test_split_sample(samples, test_size)
    return dict(train=train_samples, validation=test_samples)




