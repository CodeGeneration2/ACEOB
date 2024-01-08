# -*- coding: utf-8 -*-
# print(f":\n{}")
# ############################################################################################################################################

import re
import types
import os
import re
from tqdm import tqdm

import types
import ast, astunparse

import ast, astunparse

import astor


# #################################################### Function to Clean Variable Names ##################################################
def clean_variable_names(AST_model):
    # #################################################### Looping ####################################################
    # ------------------- Auxiliary Tools -------------#
    variable_count = 0
    function_count = 0
    renamed_variable_dict = {}
    renamed_function_dict = {}
    for some_node in ast.walk(AST_model):
        # ----------------------------------------- If it's a function argument declaration -------------------#
        if isinstance(some_node, ast.arg):
            # ------------------------------- Handling named and unnamed variables -----------------#
            if some_node.arg in renamed_variable_dict:
                some_node.arg = renamed_variable_dict[some_node.arg]
            else:
                variable_count += 1
                renamed_variable_dict[some_node.arg] = f"variable_{variable_count}"
                some_node.arg = f"variable_{variable_count}"
        # ----------------------------------------- If it's a normal variable declaration -------------------#
        if isinstance(some_node, ast.Name):
            # ------------------------------- Handling named and unnamed variables and reserved words -----------------#
            if some_node.id in renamed_variable_dict:
                some_node.id = renamed_variable_dict[some_node.id]
            elif isinstance(some_node.ctx, ast.Load):
                continue
            else:
                variable_count += 1
                renamed_variable_dict[some_node.id] = f"variable_{variable_count}"
                some_node.id = f"variable_{variable_count}"
        # ----------------------------------------- If it's a function declaration -------------------#
        if isinstance(some_node, ast.FunctionDef):
            function_count += 1
            renamed_function_dict[some_node.name] = f"function{function_count}"
            some_node.name = f"function{function_count}"

    # ============================================== Second Loop to fix any missing parts ====================================#
    for some_node in ast.walk(AST_model):
        # ----------------------------------------- If it's a function argument declaration -------------------#
        if isinstance(some_node, ast.arg):
            # ------------------------------- Handling named variables -----------------#
            if some_node.arg in renamed_variable_dict:
                some_node.arg = renamed_variable_dict[some_node.arg]

        # ----------------------------------------- If it's a normal variable declaration -------------------#
        if isinstance(some_node, ast.Name):
            # ------------------------------ Handling named variables -----------------#
            if some_node.id in renamed_variable_dict:
                some_node.id = renamed_variable_dict[some_node.id]
            # ------------------------------ Handling named functions -----------------#
            if some_node.id in renamed_function_dict:
                some_node.id = renamed_function_dict[some_node.id]


    messy_code = astunparse.unparse(AST_model)
    # ============================================== First step: marking the first occurrence of each variable ============================#
    total_variables = variable_count
    variable_positions = []
    variable_position_dict = {}
    for variable in range(1, total_variables+1):
        variable_position = messy_code.index(f"variable_{variable}")
        variable_positions.append(variable_position)
        variable_position_dict[variable_position] = f"variable_{variable}"

    # ============================================== Second Step: Creating a correction dictionary ==============================#
    correction_dict = {}  # Format: "old_variable_name": "new_variable_name"
    sorted_variable_positions = variable_positions.copy()
    sorted_variable_positions.sort(reverse=True)

    current_variable_count = total_variables
    for position in sorted_variable_positions:
        new_variable_name = f"var{current_variable_count}"
        old_variable_name = variable_position_dict[position]
        correction_dict[old_variable_name] = new_variable_name
        current_variable_count -= 1

    # ============================================== Third Step: Start correcting ====================================#
    for some_node in ast.walk(AST_model):
        # ----------------------------------------- If it's a function argument declaration -------------------#
        if isinstance(some_node, ast.arg):
            # ------------------------------- Handling named variables -----------------#
            if some_node.arg in correction_dict:
                some_node.arg = correction_dict[some_node.arg]

        # ----------------------------------------- If it's a normal variable declaration -------------------#
        if isinstance(some_node, ast.Name):
            # ------------------------------ Handling named variables -----------------#
            if some_node.id in correction_dict:
                some_node.id = correction_dict[some_node.id]

    return AST_model







# ################################################################ AST Code Purification #########################################
def purify_AST_code(code_string):
    """
    1, Import code
    2, Convert to AST and back
    3, Delete the original code, replace with the new one
    """
    
    code_string = code_string.strip()
    # ---------------------------- Convert to AST and back -----------------------#
    try:
        AST_model = ast.parse(code_string)
        AST_model = clean_variable_names(AST_model)
        purified_code = astor.to_source(AST_model)
    except:
        return -1

    return purified_code






# ############################################################################################################################################
if __name__ == '__main__':
    dataset_path = "F:/ECG Dataset"
    junkyard_path = "F:/Junkyard/Junk1"
    generated_set_path = "F:/ECG_CG Dataset"


    code_path = f"./127,109 ms,20 KB,Standard time 96 ms,NPI 9800.txt"

    # ============================================== Import Code ===================#
    with open(code_path, 'r', encoding='UTF-8') as f:
        code_string = f.read()
    code_string = code_string.strip()


    purified_code = purify_AST_code(code_string)

    # --------------------------------------------- Delete the original code, replace with new ---------------------#
    os.remove(code_path)
    with open(code_path, 'w', encoding='UTF-8') as f:
        f.write(purified_code.strip())


