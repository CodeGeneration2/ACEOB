'\nDataset to be used for APPS Training\n'
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

class Train_Dataset(torch.utils.data.Dataset):

    def __init__(self, Data_set_root_path='ECG'):
        self.Tokenized_GPT = transformers.GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')
        self.Total_Data_List = []
        self.Initialize(f'{Data_set_root_path}/train')

    def Initialize(self, Data_set_root_path):
        Total_data_list = []
        List_of_training_sets = os.listdir(f'{Data_set_root_path}')
        for certain_entry in tqdm(range(len(List_of_training_sets))):
            with open(f'{Data_set_root_path}/{certain_entry}/Accepted.txt', 'r', encoding='UTF-8') as f:
                The_tag_code = f.read()
            Tag_code_tensor_dictionary = self.Tokenized_GPT(The_tag_code, return_tensors='pt')
            if (len(Tag_code_tensor_dictionary['input_ids'][0]) > 766):
                continue
            The_tag_code = Indent_code_functions(The_tag_code)
            with open(f'{Data_set_root_path}/{certain_entry}/Title.txt', 'r', encoding='UTF-8') as f:
                Title = f.read()
            with open(f'{Data_set_root_path}/{certain_entry}/Problem description body.txt', 'r', encoding='UTF-8') as f:
                Body_of_the_problem_description = f.read()
            with open(f'{Data_set_root_path}/{certain_entry}/Input description.txt', 'r', encoding='UTF-8') as f:
                Input_description = f.read()
            with open(f'{Data_set_root_path}/{certain_entry}/Output description.txt', 'r', encoding='UTF-8') as f:
                Output_description = f.read()
            with open(f'{Data_set_root_path}/{certain_entry}/I_O sample tests and Note description.txt', 'r', encoding='UTF-8') as f:
                Input_Output_Sample_Tests_and_Note_Descriptions = f.read()
            List_of_slow_code_sets = os.listdir(f'{Data_set_root_path}/{certain_entry}/Acc_tle_solutions')
            for certain_code in List_of_slow_code_sets:
                with open(f'{Data_set_root_path}/{certain_entry}/Acc_tle_solutions/{certain_code}', 'r', encoding='UTF-8') as f:
                    slow_code = f.read()
                slow_code = Indent_code_functions(slow_code)
                slow_code_tensor_dictionary = self.Tokenized_GPT(slow_code, return_tensors='pt')
                if (len(slow_code_tensor_dictionary['input_ids'][0]) > 768):
                    continue
                Input_and_output_case_paths = f'{Data_set_root_path}/{certain_entry}/IO Case Test Dictionary.txt'
                Data_original_path = f'{Data_set_root_path}/{certain_entry}/acc_tle_solutions/{certain_code}'
                certain_data_tuple = (Title, Body_of_the_problem_description, Input_description, Output_description, Input_Output_Sample_Tests_and_Note_Descriptions, slow_code, The_tag_code, Input_and_output_case_paths, Data_original_path)
                Total_data_list.append(certain_data_tuple)
        self.Total_Data_List = Total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, Index):
        Sample_list = self.Total_Data_List[Index]
        return Sample_list

def Indent_code_functions(Code_string):
    Code_string = io.StringIO(Code_string)
    Code_string_after_indentation = io.StringIO()
    run_reindent(Code_string, Code_string_after_indentation, config={'dry-run': False, 'help': False, 'to': 4, 'from': (- 1), 'tabs': True, 'encoding': 'utf-8', 'is-tabs': False, 'tabsize': 4, 'all-tabs': False})
    return Code_string_after_indentation.getvalue()