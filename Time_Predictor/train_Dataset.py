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
        for particular in tqdm(range(len(List_of_training_sets))):
            with open(f'{Data_set_root_path}/{particular}/Accepted.txt', 'r', encoding='UTF-8') as f:
                Tag_Code = f.read()
            Tag_code_tensor_dictionary = self.Tokenized_GPT(Tag_Code, return_tensors='pt')
            if (len(Tag_code_tensor_dictionary['input_ids'][0]) > 766):
                continue
            Tag_Code = Indentation_code_functions(Tag_Code)
            with open(f'{Data_set_root_path}/{particular}/Accepted run time.txt', 'r', encoding='UTF-8') as f:
                Run_time = f.read().split('accepted,')[1].split(' ms,')[0]
            Run_time = int(Run_time)
            certain_data_tuple = (Tag_Code, Run_time)
            Total_data_list.append(certain_data_tuple)
            List_of_slow_code_sets = os.listdir(f'{Data_set_root_path}/{particular}/Acc_tle_solutions')
            for certain_code in List_of_slow_code_sets:
                with open(f'{Data_set_root_path}/{particular}/Acc_tle_solutions/{certain_code}', 'r', encoding='UTF-8') as f:
                    slow_code = f.read()
                slow_code = Indentation_code_functions(slow_code)
                slow_code_tensor_dictionary = self.Tokenized_GPT(slow_code, return_tensors='pt')
                if (len(slow_code_tensor_dictionary['input_ids'][0]) > 768):
                    continue
                slow_code = Indentation_code_functions(slow_code)
                Run_time = int(certain_code.split(',')[1].split(' ms')[0])
                certain_data_tuple = (slow_code, Run_time)
                Total_data_list.append(certain_data_tuple)
            List_of_better_codesets = os.listdir(f'{Data_set_root_path}/{particular}/Acc_solutions')
            for certain_code in List_of_better_codesets:
                with open(f'{Data_set_root_path}/{particular}/Acc_solutions/{certain_code}', 'r', encoding='UTF-8') as f:
                    better_code = f.read()
                better_code = Indentation_code_functions(better_code)
                better_code_tensor_dictionary = self.Tokenized_GPT(better_code, return_tensors='pt')
                if (len(better_code_tensor_dictionary['input_ids'][0]) > 768):
                    continue
                better_code = Indentation_code_functions(better_code)
                Run_time = int(certain_code.split(',')[1].split(' ms')[0])
                certain_data_tuple = (better_code, Run_time)
                Total_data_list.append(certain_data_tuple)
        self.Total_Data_List = Total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, Index):
        Sample_list = self.Total_Data_List[Index]
        return Sample_list

def Indentation_code_functions(Code_string):
    Code_string = io.StringIO(Code_string)
    Indented_code_strings = io.StringIO()
    run_reindent(Code_string, Indented_code_strings, config={'dry-run': False, 'help': False, 'to': 4, 'from': (- 1), 'tabs': True, 'encoding': 'utf-8', 'is-tabs': False, 'tabsize': 4, 'all-tabs': False})
    return Indented_code_strings.getvalue()