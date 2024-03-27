from IPython.display import HTML, display, Markdown

def show_data_point(data_point):
    def replace_trans(text):
        text = text.replace('\\n', '\n')
        return text
    keys = list(data_point.keys())
    for key in keys:
        print(key + ":")
        value = data_point[key]
        if isinstance(value, str):
            print(replace_trans(data_point[key]))
        else:
            print(value)
        print("-"*20)

def get_pos(pos_str):
    # 'i_j' => [i, j]
    return [int(r) for r in pos_str.split('_')]

def check_table_list(table_list):
    col_num = len(table_list[0])
    for r in table_list:
        assert len(r) == col_num

def preprocess_cell_text(text: str):
    # preprocess table cells
    text = text.strip()
    return text.replace('\n', '')

def convert_table_dict_to_table_list(table_dict: dict, merged=[], mark_cell_type=False, debug=False):
    """
    @param table_dict: cell_pos2cell_content
    @return table: a list of rows. For merged cells, duplicate the non-empty merged cells.
    """
    cell_pos = [get_pos(k) for k in table_dict.keys()]
    max_row = max([p[0] for p in cell_pos])
    max_col = max([p[1] for p in cell_pos])
    table = [['' for _ in range(max_col+1)] for _ in range(max_row+1)]
    for k, v in table_dict.items():
        r, c = get_pos(k)
        cell_content = preprocess_cell_text(v['text'])
        is_head2mark = {'row': '(rh)', 'col': '(ch)'}
        if mark_cell_type:
            cell_content = cell_content + is_head2mark.get(v['is_head'], '')
        table[r][c] = cell_content
    
    for mcells in merged:
        non_empty_mcells = []
        for cell in mcells:
            if table[cell[0]][cell[1]] != '':
               non_empty_mcells.append(cell)
        if len(non_empty_mcells) > 1:
            if debug:
                print("warning: multi non-empty cells in merged cells")
            raise ValueError("multi cells")
            dup_cell = non_empty_mcells[0]
        elif  len(non_empty_mcells) == 0:
            if debug:
                print("warning: empty non-empty cells")
            raise ValueError("no cells")
            dup_cell = mcells[0]
        else:
            dup_cell = non_empty_mcells[0]
        
        for cell in mcells:
            table[cell[0]][cell[1]] = table[dup_cell[0]][dup_cell[1]]
    check_table_list(table)
    return table

def convert_table_list_to_md_str(table_list):
    """
    convert a table (list of list of str) to its markdown format.
    """
    def convert_one_row(cells):
        cells = [preprocess_cell_text(c) for c in cells]
        cell_strs = [f"| {c} " for c in cells]
        row_str = ''.join(cell_strs) + '|\n'
        return row_str
    assert len(table_list) > 1
    check_table_list(table_list)
    header = convert_one_row(table_list[0])
    split = convert_one_row(['-' for _ in range(len(table_list[0]))])
    md_str = header + split
    for row in table_list[1:]:
        md_str += convert_one_row(row)
    return md_str

def clean_cell_dict(cells):
    new_cells = {}
    for cell_pos, cell_info in cells.items():
        new_cell_info = dict(text=cell_info['text'], is_header=cell_info['is_head'])
        new_cells[cell_pos] = new_cell_info
    return new_cells

def display_table(table):
    table_md = convert_table_list_to_md_str(table)
    display(Markdown(table_md))