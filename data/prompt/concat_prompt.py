
def get_concat_template():
    template = "{task_description}\n\n{demo_qa}\n\n{input_q}"
    return template

def get_qa_template(sep_qa=False, has_caption=False, has_passage=False, has_options=False):
    # examples: qa_template = "Q: {question}\nTable:\n{table}\nA: {answer}"
    # select template
    qa_template = "Question: {question}"
    if has_caption:
        qa_template += "\nMeta Information:\n{caption}"
    # the meta information should be ahead of the table
    qa_template += "\nTable:\n{table}"
    if has_passage:
        qa_template += "\nPassage:\n{passage}"
    if has_options:
        qa_template += "\nOptions:\n{options}"
    qa_template += "\n"
    if sep_qa:
        return qa_template, "{answer}"
    else:
        qa_template += "Answer: {answer}"
        return qa_template