import os
from typing import Union
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from IPython.display import HTML, display, Markdown
from tab_benchmark.utils import default_dump_md


class Table():
    """
    Table class for Table Question Answering. Supporting merged cells.
    Basic data structure: List[List[dict{'content', 'rowspan', 'colspan', 'is_header'}]]
    @attr meta: Optional[Dict, str]. Meta information of the table, such as section_title and caption.
                If not None, must contain a key: `summary`. The `summary` is applied in the final prompt as the Caption. 
                If `str` is given, it will be automatically transformed to a dict with key `summary`.
    """
    def __init__(self, table_data, table_type='rel', meta=None) -> None:
        self.table_data = self.preprocess(table_data)
        self.table_type = table_type

        if meta is not None:
            if isinstance(meta, dict):
                assert 'summary' in meta
            elif isinstance(meta, str):
                meta = dict(summary=meta)
            else:
                raise TypeError(f"meta must be Dict or str.")
        self.meta = meta

        # self.bad_table = False  # TODO: raise error?
        self.row_num, self.col_num = self.grid_size()
        # print(self.row_num, self.col_num)
        self.grid_table = self.get_grid_table()
        self.md_table = Table.convert_table_data_to_md_str(self.grid_table  )

    def preprocess(self, table_data):
        # Preprocess cell content in the table.
        for row in table_data:
            for cell in row:
                cell["content"] = Table.preprocess_cell_text(cell["content"])
        return table_data
    
    def grid_size(self):
        row_num = len(self.table_data)
        # some table are handled badly, Maybe the first row is not the widest,such as ['train'][3865]
        col_num = [sum([c.get("colspan", 1) for c in self.table_data[i]]) for i in range((row_num))]
        col_num = sorted(col_num)
        # if col_num[0] != col_num[-1]:
        #     self.bad_table = True
        # print(row_num, col_num[0])
        return row_num, max(col_num)
    
    def get_grid_table(self):
        # split merged cells and convert the table_data to grid table (List[List[cell]])
        # one merged cell is copied to split cells of the merged cells.
        grid_table = [[None for _ in range(self.col_num)] for _ in range(self.row_num)]
        # if self.bad_table:
        #     print()
        # assert self.bad_table, "bad table?"
        for ridx, row in enumerate(self.table_data):
            cidx = 0
            for cell in row:
                # print(grid_table)
                # some cell has no content but actually not exist in original table, such as ['train'][353]
                # if cell.get('content', None) == None:
                #     continue
                # if ridx >= self.row_num or cidx >= self.col_num:
                #     self.bad_table = True
                #     return grid_table
                while grid_table[ridx][cidx] is not None:
                    cidx += 1
                    if ridx >= self.row_num or cidx >= self.col_num:
                        self.bad_table = True
                        assert self.bad_table, "beyond column number"
                        return grid_table
                rowspan = cell.get("rowspan", 1)
                colspan = cell.get("colspan", 1)
                # some cell have a very large rowspan or colspan, such as ['train'][353]
                for i in range(rowspan):
                    if ridx + i >= self.row_num:
                        break
                    # grid_table[ridx+i][cidx] = cell["content"]
                    for j in range(colspan):
                        if cidx + j >= self.col_num:
                            assert self.bad_table, "beyond column number"
                            break
                        grid_table[ridx+i][cidx+j] = cell["content"]
                
        return grid_table

    def display(self):
        table_md = Table.convert_table_data_to_md_str(self.grid_table)
        display(Markdown(table_md))
        if self.meta:
            print(f"table meta info: {self.meta}")

    @staticmethod
    def preprocess_cell_text(text: str):
        # preprocess table cells
        text = text.strip() 
        return text.replace('\n', '')

    @staticmethod
    def from_header_rows(headers, rows, table_type = 'rel', meta=None) -> 'Table':
        grid_table = [headers, *rows]
        table_type = 'rel'
        return Table.from_grid_table(grid_table, table_type=table_type, meta=meta)

    @staticmethod
    def from_grid_table(grid_table, table_type='rel', meta=None) -> 'Table':
        table_data = [
            [{'content': cell, 'rowspan': 1, 'colspan': 1, 'is_header': None} for cell in row]
            for row in grid_table]
        return Table(table_data=table_data, table_type=table_type, meta=meta)

    @staticmethod
    def transpose_grid_table(grid_table):
        # transpose rows and columns of a grid table (List[List[cell]])
        return [list(i) for i in zip(*grid_table)]

    @staticmethod
    def convert_table_data_to_md_str(grid_table):
        """
        convert grid table to a markdown string.
        """
        # List[List[]]
        # convert a table list to a markdown table
        def convert_one_row(cells, is_last_row=False):
            cells = [c for c in cells]
            cell_strs = [f"| {c} " for c in cells]
            if is_last_row:
                row_str = ''.join(cell_strs) + '|'
            else:
                row_str = ''.join(cell_strs) + '|\n'
            return row_str
        assert len(grid_table) >= 1
        header = convert_one_row(grid_table[0])
        split = convert_one_row(['---' for _ in range(len(grid_table[0]))])
        md_str = header + split
        # md_str = header
        for row in grid_table[1:-1]:
            md_str += convert_one_row(row)
        md_str += convert_one_row(grid_table[-1], is_last_row=True)
        return md_str
    
    @staticmethod
    def convert_table_data_to_html_str(table_data):
        """
        convert table data to a html string.
        """
        html = '<table>\n'
        for row in table_data:
            html += '<tr>\n'
            for cell in row:
                if cell.get('is_header', False):
                    html += '<th'
                else:
                    html += '<td'
                if 'rowspan' in cell and cell['rowspan'] > 1:
                    html += ' rowspan="{}"'.format(cell['rowspan'])
                if 'colspan' in cell and cell['colspan'] > 1:
                    html += ' colspan="{}"'.format(cell['colspan'])
                html += '>'
                html += cell['content']
                if cell.get('is_header', False):
                    html += '</th>\n'
                else:
                    html += '</td>\n'
            html += '</tr>\n'
        html += '</table>'
        return html


class OldTable():
    available_table_types = ['rel']
    def __init__(self, table, meta=None, table_type='rel'):
        # @param table: List[List[]]
        # @param meta: str, caption/table info
        # @param table_type: ['rel',]  TODO
        assert table_type in Table.available_table_types
        self.table = table
        self.meta = meta
        Table.check_table_list(self.table)
        self.table_type = table_type

        self.row_num, self.col_num = Table.get_col_row_num(table)
        self.preprocess()

    def preprocess(self):
        for row in self.table:
            for ele in row:
                ele = Table.preprocess_cell_text(ele)

    def display(self):
        table_md = Table.convert_table_list_to_md_str(self.table)
        display(Markdown(table_md))
        if self.meta:
            print(f"table meta info: {self.meta}")

    @staticmethod
    def from_header_rows(headers, rows, meta=None) -> 'Table':
        table = [headers, *rows]
        table_type = 'rel'
        return Table(table, table_type=table_type, meta=meta)

    @staticmethod
    def check_table_list(table_list: list):
        # check if the table list is valid 
        assert len(table_list) > 0 and len(table_list[0]) > 0
        col_num = len(table_list[0])
        for r in table_list:
            assert len(r) == col_num

    @staticmethod
    def preprocess_cell_text(text: str):
        # preprocess table cells
        text = text.strip()
        return text.replace('\n', '')

    @staticmethod
    def convert_table_list_to_md_str(table_list):
        # List[List[]]
        # convert a table list to a markdown table
        def convert_one_row(cells, is_last_row=False):
            cells = [Table.preprocess_cell_text(c) for c in cells]
            cell_strs = [f"| {c} " for c in cells]
            if is_last_row:
                row_str = ''.join(cell_strs) + '|'
            else:
                row_str = ''.join(cell_strs) + '|\n'
            return row_str
        assert len(table_list) >= 1
        Table.check_table_list(table_list)
        header = convert_one_row(table_list[0])
        split = convert_one_row(['---' for _ in range(len(table_list[0]))])
        md_str = header + split
        # md_str = header
        for row in table_list[1:-1]:
            md_str += convert_one_row(row)
        md_str += convert_one_row(table_list[-1], is_last_row=True)
        return md_str
    
    @staticmethod
    def convert_table_list_to_html_str(table_list):
        html = "<table>\n"

        for row in table_list:
            html += "<tr>"
            for cell in row:
                html += "<td>{}</td>".format(cell)
            html += "</tr>\n"

        html += "</table>"
        return html

    @staticmethod
    def get_col_row_num(table):
        return len(table), len(table[0])
    

@dataclass
class TableQASample():
    task_type: str
    table: Table
    question: str
    answer: Union[str, list]
    source: str
    passage: Optional[str] = None
    options: Optional[Union[str, list]] = None
    option_types: Optional[list] = None
    evidence: Optional[list] = None
    other_info: Optional[dict] = None

    def __post_init__(self):
        if self.other_info is None:
            self.other_info = {}
        self.other_info["question"] = self.question
        self.other_info["grid_table"] = self.table.grid_table

    def show(self):
        self.table.display()
        print(f"task_type: {self.task_type} | source: {self.source}")
        if self.passage:
            print(f"passage: {self.passage}")
        print(f"question: {self.question}")
        print(f"answer: {self.answer}")
        if self.options:
            print(f"options: {self.options}")
        if self.evidence:
            print(f"evidence: {self.evidence}")

    @staticmethod
    def dump_check_option_md(tqa_dataset: List['TableQASample'], option_dump_path: str):
        """
        To check the generated option, save each TQAsample to a markdown file.
        """

        md_files = []
        tidx = 0
        for idx, sample in enumerate(tqa_dataset):
            if sample.options is None:
                continue
            tidx += 1
            md_files.append(f"## sample {tidx}-{idx}\n")
            md_files.append(f"Question: {sample.question}\n")
            md_files.append(f"Table:\n")
            md_files.append(Table.convert_table_data_to_md_str(sample.table.grid_table))

            if sample.table.meta is not None:
                 md_files.append(f"Meta Information: {sample.table.meta}\n")
            
            if sample.passage is not None:
                md_files.append(f"Passage: {sample.passage}\n")
            
            md_files.append("\n")
            md_files.append(f"Options:\n{sample.options}\n")
            md_files.append(f"Answer: {sample.answer}\n")
            md_files.append(f"Score:\n")
            md_files.append(f"Comment:\n")
        
        default_dump_md(md_files, os.path.join(option_dump_path, "check_option.md"))

