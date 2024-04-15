# coding=utf-8

from statistics import mean

import torch

import os

from tqdm import tqdm
from transformers import RobertaTokenizer, AutoTokenizer
from Code_to_AST.Code_to_AST import code_to_AST
import random
import json


# ################################################################################################
class MyDatasetFunction(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_token_count=512, model_path='codet5-base', train_or_predict=""):

        self.dataset_path = dataset_path
        self.max_token_count = max_token_count
        self.model_path = model_path
        self.train_or_predict = train_or_predict

        # ------------------------------------ Tokenization Vocabulary ------------------------------------#
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.total_data_list = []  # Should be set in the initialize function
        self.total_data_dict = {}  # Should be set in the initialize function

        # ========================= Initialize Function (Import Data Locally) ===============#
        self.initialize()

    # =========================================== Initialize Function (Import Data Locally) =========================================#
    def initialize(self):
        """
        Import data locally
        Returns:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_path is set to folderName/data
        """

        total_data_list = []

        # ============================================================= Features ====================================#
        code_list = os.listdir(f"{self.dataset_path}")
        for code_file in code_list:
            with open(f"{self.dataset_path}/{code_file}", 'r', encoding='UTF-8') as f:
                code = f.read()
            AST = code_to_AST(code)
            input_feature = f"{AST}\nEvaluate the code running time:"

            running_time = str(int(code_file.split(",Running time ")[-1].split(" ms,")[0]))
            # running_time = str(float(code_file.split(" ms,NPI ")[-1].split(",")[0]))

            # ---------------------------------------- Minimal Unit ----------------------------------------#
            data_tuple = (input_feature, running_time, f"{self.dataset_path}/{code_file}")
            # -------------------------- Add to Total Data List -----------#
            total_data_list.append(data_tuple)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== {len(total_data_list)} training set data loaded ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iterator Function ==================================#
    def __getitem__(self, index):
        input_feature, running_time, code_path = self.total_data_list[index]

        # ---------------------------- Never to be deleted. Delay bug fixes. --------------#
        input_feature = input_feature[:150000]
        running_time = running_time[:150000]

        # ------------------------------------- Encoding ---------------------------------------------#
        feature_encoding_dict = self.tokenizer(input_feature, padding='max_length', truncation=True, max_length=self.max_token_count, return_tensors="pt")

        # ------------------------------------- Encoding ---------------------------------------------#
        label_encoding_dict = self.tokenizer(running_time, padding='max_length', max_length=6, return_tensors="pt")

        if self.train_or_predict == "train":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoding_dict["input_ids"] = feature_encoding_dict['input_ids'].squeeze()
            feature_encoding_dict["attention_mask"] = feature_encoding_dict['attention_mask'].squeeze()
            feature_encoding_dict["labels"] = label_encoding_dict['input_ids'].squeeze()


        elif self.train_or_predict == "predict":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoding_dict["input_feature"] = input_feature
            feature_encoding_dict["running_time"] = running_time
            feature_encoding_dict["code_path"] = code_path

        else:
            print("Error! Check self.train_or_predict!")

        if "incoder" in self.model_path:
            feature_encoding_dict.pop("token_type_ids", None)

        return feature_encoding_dict
