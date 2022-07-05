# coding=utf-8

import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from dataset_lm.reindent import run as run_reindent
from tqdm import tqdm


import json


class Predicted_Dataset_Class(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path="Code set to be predicted"):

        self.Tokenized = transformers.GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")

        self.total_Data_List = []

        self.initialize_function_from_current_directory(dataset_root_path)


    def initialize_function_from_current_directory(self, dataset_root_path):


        total_Data_List = []

        training_list_sets = os.listdir(f"{dataset_root_path}")
        for i in tqdm(training_list_sets):
             if i.split(",")[1]=="BLEU_score":

                with open(f"{dataset_root_path}/{i}", 'r', encoding='UTF-8') as f:
                    code = f.read()


                total_Data_List.append(code)

        print(f'\033[0:35m========================== load {len(total_Data_List)} =================\033[m')

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

