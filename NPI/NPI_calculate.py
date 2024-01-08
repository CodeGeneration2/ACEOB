# -*- coding: utf-8 -*-
# print(f":\n{}")
# ############################################################################################################################################

import os
from statistics import mean

from tqdm import tqdm

from numpy import mean

import time

from tqdm import tqdm
import logging
from CodeBleu import _bleu

import difflib
from numpy import mean
import math

from AST_Purge_Reset_Variable_Function_Name import purify_AST_code

print('\033[0;34m=======================Keep Going, You Got This!==============================\033[m')


# ###############################################################################################################
test_set_root_path = r"F:\4"
dataset_path = r"F:\ACEOB\test"


# ###################################################################################################
def overall_evaluation_function(test_set_path, dataset_path):

    problem_list = os.listdir(test_set_path)
    for problem_index in tqdm(problem_list):

        various_code_pairs_list = os.listdir(f"{test_set_path}/{problem_index}")
        for code_set in various_code_pairs_list:

            if code_set.startswith("Efficient-Inefficient Code Pair"):
                
                code_list = os.listdir(f"{test_set_path}/{problem_index}/{code_set}")
                for code in code_list:   
                    
                    predicted_time = code.split(",predicted time ")[-1].split(" hr")[0]

                    try:
                        predicted_time = int(predicted_time)
                    except:
                        # print(f"{test_set_path}/{problem_index}/{code_set}/{code}")
                        continue

                    if os.path.exists(f"{dataset_path}/{problem_index}/metadata.txt"):
                        with open(f"{dataset_path}/{problem_index}/metadata.txt", 'r', encoding='UTF-8') as f:
                            time_dictionary = eval(f.read())
                    else:
                        # print(f"Label does not exist: {dataset_path}/{problem_index}")
                        continue

                    NPI = get_NPI_function(predicted_time, time_dictionary)
                    
                    new_name = code.replace(".txt", f",NPI {NPI} N.txt")

                    os.rename(f"{test_set_path}/{problem_index}/{code_set}/{code}",f"{test_set_path}/{problem_index}/{code_set}/{new_name}")



# ############################################################################################################################################
def get_NPI_function(predicted_time, time_dictionary):
    max_time = time_dictionary['Maximum time to achieve function (ms)']
    min_time = time_dictionary['Minimum time to achieve function (ms)']
    median_time = time_dictionary['Median time to achieve function (ms)']

    if predicted_time < median_time:
        NPI = 100 * (median_time - predicted_time)/(median_time - min_time)
    elif predicted_time == median_time:
        NPI = 0
    elif predicted_time > median_time:
        NPI = 100 * (median_time - predicted_time)/(max_time - median_time)

    NPI = round(NPI, 2)

    return NPI




# #####################################################################################################################################
if __name__ == '__main__':

    list_dirs = os.listdir(f"{test_set_root_path}")

    for each_set in list_dirs:
        overall_evaluation_function(test_set_path=f"{test_set_root_path}/{each_set}", dataset_path=dataset_path)
        
