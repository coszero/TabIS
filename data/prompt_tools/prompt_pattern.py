


prompt_pattern = r'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: (?P<demo_question>[\s\S]+?)\nMeta Information:\n(?P<demo_meta_info>[\s\S]+?)\nTable:\n(?P<demo_table>[\s\S]+?)\nOptions:\nA. (?P<demo_opt_a>[\s\S]+?)\nB. (?P<demo_opt_b>[\s\S]+?)\nAnswer: (?P<demo_answer>[\s\S]+?)\n\nQuestion: (?P<question>[\s\S]+?)\nMeta Information:\n(?P<meta_info>[\s\S]+?)\nTable:\n(?P<table>[\s\S]+?)\nOptions:\nA. (?P<opt_a>[\s\S]+?)\nB. (?P<opt_b>[\s\S]+?)\nAnswer: '
prompt_template = 'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: {demo_question}\nMeta Information:\n{demo_meta_info}\nTable:\n{demo_table}\nOptions:\nA. {demo_opt_a}\nB. {demo_opt_b}\nAnswer: {demo_answer}\n\nQuestion: {question}\nMeta Information:\n{meta_info}\nTable:\n{table}\nOptions:\nA. {opt_a}\nB. {opt_b}\nAnswer: '
# Some of the HiTab samples don't contain meta information.
prompt_pattern_wo_meta = r'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: (?P<demo_question>[\s\S]+?)\nMeta Information:\n(?P<demo_meta_info>[\s\S]+?)\nTable:\n(?P<demo_table>[\s\S]+?)\nOptions:\nA. (?P<demo_opt_a>[\s\S]+?)\nB. (?P<demo_opt_b>[\s\S]+?)\nAnswer: (?P<demo_answer>[\s\S]+?)\n\nQuestion: (?P<question>[\s\S]+?)\nTable:\n(?P<table>[\s\S]+?)\nOptions:\nA. (?P<opt_a>[\s\S]+?)\nB. (?P<opt_b>[\s\S]+?)\nAnswer: '
prompt_template_wo_meta = 'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: {demo_question}\nMeta Information:\n{demo_meta_info}\nTable:\n{demo_table}\nOptions:\nA. {demo_opt_a}\nB. {demo_opt_b}\nAnswer: {demo_answer}\n\nQuestion: {question}\nTable:\n{table}\nOptions:\nA. {opt_a}\nB. {opt_b}\nAnswer: '

# rag
prompt_pattern_rag = r'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: (?P<demo_question>[\s\S]+?)\nMeta Information:\n(?P<demo_meta_info>[\s\S]+?)\nTable 1:\n(?P<demo_table_1>[\s\S]+?)\nTable 2:\n(?P<demo_table_2>[\s\S]+?)\nOptions:\nA. (?P<demo_opt_a>[\s\S]+?)\nB. (?P<demo_opt_b>[\s\S]+?)\nAnswer: (?P<demo_answer>[\s\S]+?)\n\nQuestion: (?P<question>[\s\S]+?)\nMeta Information:\n(?P<meta_info>[\s\S]+?)\nTable 1:\n(?P<table_1>[\s\S]+?)\nTable 2:\n(?P<table_2>[\s\S]+?)\nOptions:\nA. (?P<opt_a>[\s\S]+?)\nB. (?P<opt_b>[\s\S]+?)\nAnswer: '
prompt_template_rag = 'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: {demo_question}\nMeta Information:\n{demo_meta_info}\nTable 1:\n{demo_table_1}\nTable 2:\n{demo_table_2}\nOptions:\nA. {demo_opt_a}\nB. {demo_opt_b}\nAnswer: {demo_answer}\n\nQuestion: {question}\nMeta Information:\n{meta_info}\nTable 1:\n{table_1}\nTable 2:\n{table_2}\nOptions:\nA. {opt_a}\nB. {opt_b}\nAnswer: '
# Some of the HiTab samples don't contain meta information.
prompt_pattern_wo_meta_rag = r'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: (?P<demo_question>[\s\S]+?)\nMeta Information:\n(?P<demo_meta_info>[\s\S]+?)\nTable 1:\n(?P<demo_table_1>[\s\S]+?)\nTable 2:\n(?P<demo_table_2>[\s\S]+?)\nOptions:\nA. (?P<demo_opt_a>[\s\S]+?)\nB. (?P<demo_opt_b>[\s\S]+?)\nAnswer: (?P<demo_answer>[\s\S]+?)\n\nQuestion: (?P<question>[\s\S]+?)\nTable 1:\n(?P<table_1>[\s\S]+?)\nTable 2:\n(?P<table_2>[\s\S]+?)\nOptions:\nA. (?P<opt_a>[\s\S]+?)\nB. (?P<opt_b>[\s\S]+?)\nAnswer: '
prompt_template_wo_meta_rag = 'Please proceed with a table-based question answering exercise. You will be provided with a table and a corresponding question. Your task is to select the most appropriate answer from the options A or B.\n\nQuestion: {demo_question}\nMeta Information:\n{demo_meta_info}\nTable 1:\n{demo_table_1}\nTable 2:\n{demo_table_2}\nOptions:\nA. {demo_opt_a}\nB. {demo_opt_b}\nAnswer: {demo_answer}\n\nQuestion: {question}\nTable 1:\n{table_1}\nTable 2:\n{table_2}\nOptions:\nA. {opt_a}\nB. {opt_b}\nAnswer: '


def get_prompt_pattern(is_rag=False, has_meta=True):
    if not is_rag:
        # ttg and hyb
        if has_meta:
            return prompt_pattern, prompt_template
        else:
            return prompt_pattern_wo_meta, prompt_template_wo_meta
    else:
        if has_meta:
            return prompt_pattern_rag, prompt_template_rag
        else:
            return prompt_pattern_wo_meta_rag, prompt_template_wo_meta_rag