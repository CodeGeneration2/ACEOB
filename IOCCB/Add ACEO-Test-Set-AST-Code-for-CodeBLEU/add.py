# -*- coding: utf-8 -*-
import os
from CodeBleu分数 import _bleu
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

print('\033[0:34m=======================Keep going, you can do it!==============================\033[m')
import random
import shutil
from AST_Purification_Reset_Variable_Function_Name import AST_clean_code

# ############################################################################################################
# ############################################################################################################
# ############################################################################################################
if __name__ == '__main__':

    dataset_path = "F:/ACEOB/test"

    question_list = os.listdir(dataset_path)

    # question_list = question_list[61:]
    for question_index in tqdm(question_list):
        code_list = os.listdir(f"{dataset_path}/{question_index}/backup_high_efficiency_code")
        os.mkdir(f"{dataset_path}/{question_index}/backup_high_efficiency_code-AST-unified_variable")
        for code_name in code_list:
            # ---------------------------- read case ---------------------#
            with open(f"{dataset_path}/{question_index}/backup_high_efficiency_code/{code_name}", 'r', encoding='UTF-8') as f:
                code = f.read()

            unified_code = AST_clean_code(code)

            # ---------------------------- case ---------------------#
            with open(f"{dataset_path}/{question_index}/backup_high_efficiency_code-AST-unified_variable/{code_name}", 'w', encoding='UTF-8') as f:
                f.write(unified_code)
