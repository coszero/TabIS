import random
import string
from typing import List


def replace_str_to_random_numbers(string, fix_length=None):
    length = fix_length or len(string) % 8 + 1
    # Generating a random number string of the same length
    random_numbers = ''.join(random.choices('0123456789', k=length))
    return random_numbers

def replace_str_to_random_lowercase(string_):
    length = len(string_) % 8 + 1
    random_numbers = ''.join(random.choices(string.ascii_lowercase, k=length))
    return random_numbers

def generate_synthetic_table_based_real_table(table: List[List[str]], transform_cell_func):
    new_table = []
    for row in table:
        new_row = []
        for cell in row:
            new_cell = transform_cell_func(cell)
            new_row.append(new_cell)
        new_table.append(new_row)
    return new_table