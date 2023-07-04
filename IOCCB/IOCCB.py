# coding=utf-8

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


def main_evaluation_function(testset_path, dataset_path):
    problem_list = os.listdir(testset_path)
    for problem_index in tqdm(problem_list):
        various_code_pairs_list = os.listdir(f"{testset_path}/{problem_index}")
        for some_code_set in various_code_pairs_list:
            some_code_list = os.listdir(f"{testset_path}/{problem_index}/{some_code_set}")
            for some_code in some_code_list:

                if "IOOCB" in some_code:
                    continue

                original_backup_code_set_path = f"{dataset_path}/{problem_index}/backup_efficient_code"
                original_CodeBLEU_Max_list = evaluation_function(f"{testset_path}/{problem_index}/{some_code_set}/{some_code}", original_backup_code_set_path)
                original_mean = mean(original_CodeBLEU_Max_list)

                unified_backup_code_set_path = f"{dataset_path}/{problem_index}/backup_efficient_code-AST-unified_variables"
                unified_CodeBLEU_Max_list = evaluation_function(f"{testset_path}/{problem_index}/{some_code_set}/{some_code}", unified_backup_code_set_path)

                unified_mean = mean(unified_CodeBLEU_Max_list)
                unified_max = max(unified_CodeBLEU_Max_list)

                IOOCB = unified_max - (unified_mean - original_mean)
                IOOCB = round(IOOCB, 2)

                new_name = some_code.replace(".txt", f",IOOCB {IOOCB} IOOCB.txt")
                os.rename(f"{testset_path}/{problem_index}/{some_code_set}/{some_code}",f"{testset_path}/{problem_index}/{some_code_set}/{new_name}")



def evaluation_function(code_path, backup_code_set_path):
    # ---------------------------- Case Reading ---------------------#
    with open(f"{code_path}", 'r', encoding='UTF-8') as f:
        tested_code = f.read()

    backup_code_list = os.listdir(f"{backup_code_set_path}")

    CodeBLEU_Max_list = []
    for backup_code_name in backup_code_list:
        # ---------------------------- Case Reading ---------------------#
        with open(f"{backup_code_set_path}/{backup_code_name}", 'r', encoding='UTF-8') as f:
            backup_code = f.read()
        try:
            Codebleu_score = round(_bleu(backup_code, tested_code), 2)
        except:
            print(f"Error: Code Path: {code_path}")
            print(f"Code Path: {tested_code}")
            print(f"Error: Backup Code Path: {backup_code_set_path}/{backup_code_name}")
            print(f"Backup Code: {backup_code}")
            Codebleu_score = 0
        CodeBLEU_Max_list.append(Codebleu_score)

    return CodeBLEU_Max_list




# ############################################################ Entry Point ####################################################
if __name__ == '__main__':
    # dataset_path = "../../43-ACEO/test"
    # generated_code_path = "./allcode"
    dataset_path = "../../test"
    generated_code_path = "../../3rd_floor_code"
    generated_code_list = os.listdir(generated_code_path)
    for some_generated_code_set in tqdm(generated_code_list):
        print(some_generated_code_set)
        main_evaluation_function(testset_path=f"{generated_code_path}/{some_generated_code_set}", dataset_path=dataset_path)
