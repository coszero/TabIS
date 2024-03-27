
exam_meta_prompt = """
## Instruction

Given a table-related task (Task), an example of the task (Example) and one input (Input), your task is to follow the task instruction and provide a response (Output) to the input. Act like a weak assistant that may generate responses that are not faithful to the table fact. Don't generate incomplete responses or too long responses. Don't explain how you come up with your response.

### Task

{task_instruct}

### Example

{demo}

### New Input

{input}

### Answers
"""

judge_meta_prompt = """
## Instruction

Given a table and a list of statements, your task is to identify which of these statements are unfaithful to the table and its meta information.  Please note that the meta information may offer additional context about the table, such as background information about the person, album, or competetion the table pertains to. Your response should in json format: {{"reasoning": your judgement of each statement, "unfaithful statements": the list of the serial number of unfaithful statements}}. Make sure your response can be parsed by json.loads.

### Table

Meta Information of the table: {meta_info}

{md_table}

### Statements

{statements}

## Response

"""

modify_answer_prompt = """\
## Instruction

You are a helpful assistant in generating one unfaithful statement. You can refer to the given faithful statement and make up a new statement that contains several highlighted cells, but is not faithful to the table fact. Basically, it is hard for a person to find that your generated statement is not faithful. Your response should in json format: {{"reasoning": your reasoning process, "unfaithful statement": the unfaithful statement}}. Make sure your response can be parsed by json.loads.

### Table

Meta Information of the table: {meta_info}

{md_table}

### Highlighted Cells

{highlighted_cells}

### Faithful Statement

{output}

## Response

"""

modify_input_prompt = """
## Instruction

You are a helpful assistant in generating one statement that is unfaithful to the table fact. Given a statement generation task, and one input-output pair of the task, you need to (1) slightly modify the input; (2) perform the task on the modified input to get the unfaithful statement. Basically, it is hard for a person to find that your generated statement is actually not faithful. Your response should in json format: {{"reasoning": Your modification of input, "unfaithful statement": the unfaithful statement}}. Make sure your response can be parsed by json.loads.

### Task

{task_instruct}

### Input

{input}

### Standard Answer

{output}

## Response

"""
