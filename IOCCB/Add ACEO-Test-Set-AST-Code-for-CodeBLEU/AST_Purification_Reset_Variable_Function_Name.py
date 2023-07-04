# coding = UTF-8

import re
import types
import os
import re
from tqdm import tqdm

import types
import ast, astunparse

import ast, astunparse

import astor

print('\033[0:33m=======================Keep going, you can do it!==============================\033[m')


# #################################################### Cleaning variable name function ##################################################
def clean_variable_name_function(AST_model):
    variable_count = 0
    function_count = 0
    renamed_variable_dict = {}
    renamed_function_dict = {}
    for node in ast.walk(AST_model):
        if isinstance(node, ast.arg):
            if node.arg in renamed_variable_dict:
                node.arg = renamed_variable_dict[node.arg]
            else:
                variable_count = variable_count + 1
                renamed_variable_dict[node.arg] = f"variable_{variable_count}"
                node.arg = f"variable_{variable_count}"
        if isinstance(node, ast.Name):
            if node.id in renamed_variable_dict:
                node.id = renamed_variable_dict[node.id]
            elif isinstance(node.ctx, ast.Load):
                continue
            else:
                variable_count = variable_count + 1
                renamed_variable_dict[node.id] = f"variable_{variable_count}"
                node.id = f"variable_{variable_count}"
        if isinstance(node, ast.FunctionDef):
            function_count = function_count + 1
            renamed_function_dict[node.name] = f"function{function_count}"
            node.name = f"function{function_count}"

    for node in ast.walk(AST_model):
        if isinstance(node, ast.arg):
            if node.arg in renamed_variable_dict:
                node.arg = renamed_variable_dict[node.arg]
        if isinstance(node, ast.Name):
            if node.id in renamed_variable_dict:
                node.id = renamed_variable_dict[node.id]
            if node.id in renamed_function_dict:
                node.id = renamed_function_dict[node.id]

    messy_variable_code = astunparse.unparse(AST_model)
    variable_total = variable_count
    variable_position_list = []
    variable_position_dict = {}
    for variable in range(1, variable_total+1):
        variable_position = messy_variable_code.index(f"variable_{variable}")
        variable_position_list.append(variable_position)
        variable_position_dict[variable_position] = f"variable_{variable}"

    correction_dict = {}
    sorted_variable_position_list = variable_position_list.copy()
    sorted_variable_position_list.sort()
    sorted_variable_position_list = sorted_variable_position_list[::-1]

    current_variable_count = variable_total
    for position in sorted_variable_position_list:
        new_variable_name = f"var{current_variable_count}"
        old_variable_name = variable_position_dict[position]
        correction_dict[old_variable_name] = new_variable_name
        current_variable_count = current_variable_count - 1

    for node in ast.walk(AST_model):
        if isinstance(node, ast.arg):
            if node.arg in correction_dict:
                node.arg = correction_dict[node.arg]
        if isinstance(node, ast.Name):
            if node.id in correction_dict:
                node.id = correction_dict[node.id]

    return AST_model


# ################################################################ AST clean code #########################################
def AST_clean_code(code_string):
    try:
        AST_model = ast.parse(code_string)
        AST_model = clean_variable_name_function(AST_model)
        cleaned_code = astor.to_source(AST_model)
    except:
        return -1

    return cleaned_code


if __name__ == '__main__':
    dataset_path = "F:/ECG"
    trash_path = "F:/"
    generate_path = "F:/ECG"

    code_path = f"./127,NPI 9800.txt"

    with open(code_path, 'r', encoding='UTF-8') as f:
        code_string = f.read()

    cleaned_code = AST_clean_code(code_string)

    os.remove(code_path)
    with open(code_path, 'w', encoding='UTF-8') as f:
        f.write(cleaned_code.strip())
