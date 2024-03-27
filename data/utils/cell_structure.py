from utils.preprocessing import check_table_list
import copy

def get_cell_from_table(table: list, i: int, j: int):
    check_table_list(table)
    row_num = len(table)
    col_num = len(table[0])
    if i < 0 or j < 0 or row_num < i or col_num < j:
        return 'None'
    else:
        return table[i][j]
    
def get_table_column_row(table: list):
    check_table_list(table)
    row_num = len(table)
    col_num = len(table[0])
    return (row_num, col_num)

def filter_tables_by_column_row(tables, row_statement, col_statement):
    indices = []
    for idx, table in enumerate(tables):
        row_num, col_num = get_table_column_row(table)
        if eval(str(row_num)+row_statement) and eval(str(col_num)+col_statement):
            indices.append(idx)
    return indices

def get_co_row_cells_from_table(table: list, cell, row_index):
    assert row_index < len(table)
    co_row = copy.deepcopy(table[row_index])
    co_row.remove(cell)
    return co_row

def get_co_col_cells_from_table(table: list, cell, col_index):
    assert col_index < len(table[0])
    co_col = copy.deepcopy([r[col_index] for r in table])
    co_col.remove(cell)
    return co_col
