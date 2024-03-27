import numpy as np
import os
import pandas as pd
import seaborn as sns
import itertools
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from tab_benchmark.utils import default_load_json
# plt.style.use('ggplot')
sns.set()

def plot_heat_map_and_line_on_different_positions(eval_base_path, line_pos=[1, 3, 5, 7], type='positional', save_path=None):
    eval_res = default_load_json(os.path.join(eval_base_path, "eval_res.json"))
    pred_res = default_load_json(os.path.join(eval_base_path, "pred_res.json"))

    print(eval_res[1])
    row_col_score = [[int(p[0].split('_')[0]), int(p[0].split('_')[1]), p[1]['em_score']] for p in eval_res[2]]
    max_row = len(set([p[0] for p in row_col_score]))
    max_col = len(set([p[1] for p in row_col_score]))
    table = [[0 for _ in range(max_col)] for _ in range(max_row)]
    for i, j, score in row_col_score:
        table[i-1][j-1] = score
    table = pd.DataFrame(table, index=[*range(1, len(table)+1)], columns=[*range(1, len(table[0])+1)])
    data = pd.DataFrame(row_col_score, columns=['row', 'column', 'EM'])
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 4, 1)
    fil_data = data.loc[data['column'].map(lambda x: x in line_pos)]
    sns.lineplot(x="row", y="EM",hue="column", data=fil_data, linewidth=2, palette=sns.color_palette())
    if save_path:
        plt.savefig(save_path+'row_em.pdf')
    # plt.show()

    plt.subplot(1, 4, 2)
    fil_data = data.loc[data['row'].map(lambda x: x in line_pos)]
    sns.lineplot(x="column", y="EM",hue="row", data=fil_data, linewidth=2, palette=sns.color_palette())
    if save_path:
        plt.savefig(save_path+'column_em.pdf')
    # plt.show()

    plt.subplot(1, 4, 3)
    plt.style.use('ggplot')
    ax = sns.heatmap(table, cmap=sns.color_palette("crest", as_cmap=True), center=50, annot=True)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # plt.xlabel('column',fontsize=10, color='k', loc='right')
    plt.title("column", fontsize=12)
    plt.ylabel('row',fontsize=12, color='k')
    if save_path:
        plt.savefig(save_path+'heatmap_all_pos.pdf')

    plt.subplot(1, 4, 4)
    if type == 'positional':
        _, _, offsets2idx = eval_pred_res(pred_res)
        sum_count = sum([len(idxs) for idxs in offsets2idx.values()])
        offset_count = [[offset[0]+1, offset[1]+1, round(len(idxs)/sum_count*100, 2)] for offset, idxs in offsets2idx.items()]
        offset_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i, j, count in offset_count:
            offset_mat[i][j] = count
        print(offset_mat)
        plt.style.use('ggplot')
        ax = sns.heatmap(offset_mat, cmap=sns.color_palette("crest", as_cmap=True), 
                        annot=True, fmt=".1f",
                        xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # plt.xlabel('column',fontsize=10, color='k', loc='right')
        plt.title("column offset", fontsize=15)
        plt.ylabel('row offset',fontsize=15, color='k')
        if save_path is not None:
            plt.savefig(save_path+'heatmap_offset.pdf')
    elif type == 'relative':
        eval_res_orient = default_load_json(os.path.join(eval_base_path, "eval_res_2.json"))
        pos_score = [[int(p[0].split('_')[0]), int(p[0].split('_')[1]), p[1]['em_score']] for p in eval_res_orient[2]]
        offset_count = [[offset_r+1, offset_c+1, score] for offset_r, offset_c, score in pos_score]
        offset_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i, j, count in offset_count:
            offset_mat[i][j] = count
        print(offset_mat)
        offset_mat = np.array(offset_mat)
        mask = offset_mat > 0
        annot = np.where(mask, offset_mat, '')
        plt.style.use('ggplot')
        ax = sns.heatmap(offset_mat, annot=annot, fmt='', cmap=mcolors.LinearSegmentedColormap.from_list("", list(zip([0.0, 1.0], ["white", "orange"]))), xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # plt.xlabel('column',fontsize=10, color='k', loc='right')
        plt.title("column offset", fontsize=12)
        plt.ylabel('row offset',fontsize=12, color='k')
        if save_path is not None:
            plt.savefig(save_path+'heatmap_offset.pdf')
    plt.show()


def get_vague_cells(table_list, tar_cell_pos, offset=1):
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

def eval_pred_res(pred_res_35):
    vague_hit_count = 0
    hit_count = 0
    offsets2idx = {}
    for idx, samp in enumerate(pred_res_35):
        table_list = samp['sample']['table_list']
        tar_cell_pos = samp['sample'].get('tar_cell_pos') or samp['sample'].get('tar_pos') 
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
    return acc, vague_acc, offsets2idx

def heatmap_compare_two_eval_res_by_substraction(compare_base_paths, titles, save_path=None):
    def get_result_table(eval_res):
        row_col_score = [[int(p[0].split('_')[0]), int(p[0].split('_')[1]), p[1]['em_score']] for p in eval_res[2]]
        max_row = len(set([p[0] for p in row_col_score]))
        max_col = len(set([p[1] for p in row_col_score]))
        table = [[0 for _ in range(max_col)] for _ in range(max_row)]
        for i, j, score in row_col_score:
            table[i-1][j-1] = score
        table = pd.DataFrame(table, index=[*range(1, len(table)+1)], columns=[*range(1, len(table[0])+1)])
        return table
    
    assert len(compare_base_paths) == len(titles), "compare_base_paths should have the same length as titles."
    
    plt.figure(figsize=(4*len(titles)+2, 4))
    for idx, (com_path, title) in enumerate(zip(compare_base_paths, titles)):
        eval_base_path_1, eval_base_path_2 = com_path
        eval_res_1 = default_load_json(os.path.join(eval_base_path_1, "eval_res.json"))
        eval_res_2 = default_load_json(os.path.join(eval_base_path_2, "eval_res.json"))

        table_1 = get_result_table(eval_res_1)
        table_2 = get_result_table(eval_res_2)

        table = table_2 - table_1
        
        plt.subplot(1, len(titles), idx+1)
        plt.style.use('ggplot')
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        if idx == len(titles) -1:
            cbar = True
        else:
            cbar = False
        ax = sns.heatmap(table, cmap=cmap, center=0, annot=True, cbar=cbar)
        # ax = sns.heatmap(table, cmap=sns.color_palette("crest", as_cmap=True), center=50, annot=True)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # plt.xlabel('column',fontsize=10, color='k', loc='right')
        ax.set_title(f"{title}\ncolumn", fontsize=10, loc='center')
        # plt.title("column", fontsize=10)
        plt.ylabel('row',fontsize=10, color='k')
    if save_path:
        plt.savefig(save_path+'heatmap_compare.pdf')
    plt.show()
    