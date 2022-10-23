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

class Prediction_Dataset(torch.utils.data.Dataset):

    def __init__(self, Data_set_root_path='Code_to_be_predicted'):
        self.Tokenized_GPT = transformers.GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')
        self.Total_Data_List = []
        self.Initialize(f'{Data_set_root_path}')

    def Initialize(self, Data_set_root_path):
        Total_data_list = []
        List_of_training_sets = os.listdir(f'{Data_set_root_path}')
        for particular in List_of_training_sets:
            with open(f'{Data_set_root_path}/{particular}', 'r', encoding='UTF-8') as f:
                Code = f.read()
            if Code:
                Total_data_list.append(Code)
        self.Total_Data_List = Total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, Index):
        Sample_list = self.Total_Data_List[Index]
        return Sample_list