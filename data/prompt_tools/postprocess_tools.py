import re

def findall_question(original_text):
    # Pattern to find the existing question
    # It looks for the string starting with 'Question:' followed by any characters until 'Meta Information:' is encountered
    pattern = r'Question:.*?(?=\nMeta Information:)'
    questions = re.findall(pattern, original_text)
    if len(questions) < 2:
        tab_pattern = r'Question:.*?(?=\nTable:)'
        questions.extend(re.findall(tab_pattern, original_text))
    return questions

    


