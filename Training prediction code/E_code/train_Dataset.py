
import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from dataset_lm.reindent import run as run_reindent
from tqdm import tqdm


import json

class TrainDatasetClass(torch.utils.data.Dataset):
    def __init__(self,  dataset_root_path="../ECG/train"):

        self.Tokenized = transformers.GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")

        self.total_Data_List = []  

        self.initialize_function_from_current_directory(dataset_root_path)

    def initialize_function_from_current_directory(self, dataset_root_path):

        total_Data_List = []

        training_list_sets = os.listdir(f"{dataset_root_path}")
        print('\033[0:34m==========================train_code================\033[m')
        for i in tqdm(range(len(training_list_sets))):

            with open(f"{dataset_root_path}/{i}/accepted.txt", 'r', encoding='UTF-8') as f:
                label_code = f.read()

            label_code_dictionary = self.Tokenized(label_code, return_tensors="pt")
            if len(label_code_dictionary["input_ids"][0])>766:
                continue

            label_code = Indent_code_functions(label_code)

            with open(f"{dataset_root_path}/{i}/Title.txt", 'r', encoding='UTF-8') as f:
                title = f.read()
            with open(f"{dataset_root_path}/{i}/Problem description body.txt", 'r', encoding='UTF-8') as f:
                problem_Description_Subject = f.read()
            with open(f"{dataset_root_path}/{i}/Input describe.txt", 'r', encoding='UTF-8') as f:
                Input_Description = f.read()
            with open(f"{dataset_root_path}/{i}/Output describe.txt", 'r', encoding='UTF-8') as f:
                Output_Description = f.read()
            with open(f"{dataset_root_path}/{i}/I_O sample tests and Note description.txt", 'r', encoding='UTF-8') as f:
                Input_output_sample_tests_and_Note_description = f.read()

            Slow_code_set_list = os.listdir(f"{dataset_root_path}/{i}/acc_tle_solutions")

            for a_code in Slow_code_set_list:
                with open(f"{dataset_root_path}/{i}/acc_tle_solutions/{a_code}", 'r', encoding='UTF-8') as f:
                    certain_slow_code = f.read()

                certain_slow_code = Indent_code_functions(certain_slow_code)
                certain_slow_code_dictionary = self.Tokenized(certain_slow_code, return_tensors="pt")
                if len(certain_slow_code_dictionary["input_ids"][0]) > 768:
                    continue

                Data_original_path = f"{dataset_root_path}/{i}/acc_tle_solutions/{a_code}"
                A_data_tuple = (title,problem_Description_Subject,Input_Description,Output_Description,Input_output_sample_tests_and_Note_description, certain_slow_code, label_code, Data_original_path)

                total_Data_List.append(A_data_tuple)

        print(f'\033[0:35m========================== load {len(total_Data_List)}  train_code =================\033[m')

        self.total_Data_List = total_Data_List


    def __len__(self):
        return len(self.total_Data_List)

    def __getitem__(self, index):

        Sample_List = self.total_Data_List[index]

        return Sample_List


def Indent_code_functions(Code_String):
    Code_String = io.StringIO(Code_String)
    Code_string_after_indentation = io.StringIO()

    run_reindent(
        Code_String,
        Code_string_after_indentation,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return Code_string_after_indentation.getvalue()


