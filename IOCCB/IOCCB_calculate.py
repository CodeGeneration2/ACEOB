# -*- coding: utf-8 -*-
# print(f":\n{}")
# ############################################################################################################################################

import os
import time

from numpy import mean


import time
# import torch
import json
import argparse
from tqdm import tqdm
import logging
from CodeBleu import _bleu

import difflib
from numpy import mean
import math

from AST_Purge_Reset_Variable_Function_Name import purify_AST_code


# ############################################################################################################################################
def total_evaluation_function(test_set_path, dataset_path):

    question_list = os.listdir(test_set_path)
    for question_index in tqdm(question_list):

        various_code_pairs_list = os.listdir(f"{test_set_path}/{question_index}")
        for code_set in various_code_pairs_list:
            
            code_list = os.listdir(f"{test_set_path}/{question_index}/{code_set}")
            for code in code_list:   
                
                if ",IOOCB " in code:
                    continue

                with open(f"{test_set_path}/{question_index}/{code_set}/{code}", 'r', encoding='UTF-8') as f:
                    original_test_code = f.read().strip()

                original_alternative_code_set_path = f"{dataset_path}/{question_index}/alternative_efficient_code"
                unified_alternative_code_set_path = f"{dataset_path}/{question_index}/alternative_efficient_code-AST-normalized_variables"

                original_CodeBLEU_Max_list = evaluation_function(original_test_code, original_alternative_code_set_path)
                
                unified_test_code = purify_AST_code(original_test_code)

                if unified_test_code == -1:
                    # print(f"unified_test_code == -1: {test_set_path}/{question_index}/{code_set}")
                    unified_CodeBLEU_Max_list = evaluation_function(original_test_code, unified_alternative_code_set_path)
                else:
                    # print(f"unified_test_code == 1: {test_set_path}/{question_index}/{code_set}")
                    unified_CodeBLEU_Max_list = evaluation_function(unified_test_code, unified_alternative_code_set_path)

                original_average = mean(original_CodeBLEU_Max_list)
                unified_average = mean(unified_CodeBLEU_Max_list)
                unified_max = max(unified_CodeBLEU_Max_list)

                mean_square_error = math.sqrt(abs(unified_average - original_average))
                if unified_average > original_average:
                    IOOCB = unified_max + mean_square_error
                else:
                    IOOCB = unified_max - mean_square_error

                if IOOCB > 100:
                    IOOCB = 100.0
                elif IOOCB < 0:
                    IOOCB = 0.0

                IOOCB = round(IOOCB, 2)

                
                # new_name = f"{code.split('N,IOOCB ')[0]}N,IOOCB {IOOCB} IC,CodeBLEU {code.split(' IC,CodeBLEU ')[-1]}"
                new_name = code.replace(".txt", f",IOOCB {IOOCB} IC.txt")

                os.rename(f"{test_set_path}/{question_index}/{code_set}/{code}",f"{test_set_path}/{question_index}/{code_set}/{new_name}")



# ############################################################################################################################################
def evaluation_function(test_code, alternative_code_set_path):

    CodeBLEU_Max_list = []

    alternative_code_list = os.listdir(f"{alternative_code_set_path}")
    for alternative_code_name in alternative_code_list:
        # ---------------------------- Reading Case ---------------------#
        with open(f"{alternative_code_set_path}/{alternative_code_name}", 'r', encoding='UTF-8') as f:
            alternative_code = f.read().strip()
        try:
            test_code = test_code.strip()
            Codebleu_score = round(_bleu(alternative_code, test_code), 2)
        except:
            #print(f"test_code: {test_code}")
            #print(f"alternative_code_path: {alternative_code_set_path}/{alternative_code_name}")
            #print(f"alternative_code: {alternative_code}")
            Codebleu_score = 0
        CodeBLEU_Max_list.append(Codebleu_score)

    return CodeBLEU_Max_list




# ############################################################ Entry Point ####################################################
if __name__ == '__main__':

    dataset_path = r"F:\0Work2：ACEOB-Code Efficiency Optimization-Dataset-SubmitJSS\0Work(2-2)-ACEOB-ReSubmitJSS\Dataset-AdjustOptimize\68-ACEOB\Test-5"
    test_set_root_path = r"F:\GeneratedCodes-5"
    list_dir = os.listdir(f"{test_set_root_path}")
    for set_dir in list_dir:
        total_evaluation_function(test_set_path=f"{test_set_root_path}/{set_dir}", dataset_path=dataset_path)
        
