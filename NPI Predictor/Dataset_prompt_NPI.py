# coding=utf-8
"""
Dataset for APPS Training
Note: Successful GPT
"""
from statistics import mean

import torch

import os

from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import AutoTokenizer

import json


# ################################################################# MyDatasetFunction #####################################
class MyDatasetFunction(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_token_num=512, model_path='codet5-base', contain_path=False):

        self.dataset_path = dataset_path
        self.max_token_num = max_token_num
        self.contain_path = contain_path

        # ------------------------------------ Tokenization vocabulary ------------------------------------#
        self.tokenization_vocabulary = RobertaTokenizer.from_pretrained(model_path)

        self.total_data_list = []  # Should be set in initialization function
        self.total_data_dict = {}  # Should be set in initialization function

        # ========================= Initialization function (import data from local) ===============#
        self.initialization_function()

    # =========================================== Initialization function (import data from local) =========================================#
    def initialization_function(self):
        """
        Import data from local
        Return:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_root_path is set to folderName/data
        """

        total_data_list = []
        # NPI_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")
        for a_problem in tqdm(dataset_list):
            # ============================================================= Features ====================================#
            code_list = os.listdir(f"{self.dataset_path}/{a_problem}")
            for a_code in code_list:
                with open(f"{self.dataset_path}/{a_problem}/{a_code}", 'r', encoding='UTF-8') as f:
                    code = f.read()

                # input_features = f"NL:{total_problem_text}\nInefficient code:{slow_code}\nEfficient code:"
                input_features = f"{code}\nEvaluate the code efficiency score:"

                NPI = int(a_code.split("ms,NPI ")[-1].split(".txt")[0])
                # NPI_list.append(NPI)

                # ---------------------------------------- Minimum unit ----------------------------------------#
                a_data_tuple = (input_features, NPI, f"{self.dataset_path}/{a_problem}/{a_code}")
                # -------------------------- Add to total data list -----------#
                total_data_list.append(a_data_tuple)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== {len(total_data_list)} train set data loaded ==================\033[m')
        # print(mean(NPI_list))

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iterator traversal function ==================================#
    def __getitem__(self, index):
        input_features, NPI, code_path = self.total_data_list[index]

        # ---------------------------- Will never be deleted. Delay bug fix. --------------#
        input_features = input_features[:150000]
        # NPI = NPI[:150000]

        # ------------------------------------- Encoding ---------------------------------------------#
        feature_encoding_dict = self.tokenization_vocabulary(input_features, padding='max_length', truncation=True, max_length=self.max_token_num,
                                        return_tensors="pt")
        # feature_encoding_dict = self.tokenization_vocabulary(input_features, padding=True, return_tensors="pt")

        # ------------------------------------- Encoding ---------------------------------------------#
        label_encoding_dict = self.tokenization_vocabulary(str(NPI), padding='max_length', max_length=6, return_tensors="pt")
        # label_encoding_dict = self.tokenization_vocabulary(str(NPI), return_tensors="pt")
        # label_tensor = label_encoding_dict['input_ids']

        feature_encoding_dict["input_ids"] = feature_encoding_dict['input_ids'][0]
        feature_encoding_dict["attention_mask"] = feature_encoding_dict['attention_mask'][0]
        feature_encoding_dict["labels"] = label_encoding_dict['input_ids'][0]
        if self.contain_path:
            feature_encoding_dict["input_features"] = input_features
            feature_encoding_dict["NPI"] = NPI
            feature_encoding_dict["code_path"] = code_path

        return feature_encoding_dict


# ==================================================================================#
# ==================================================================================#
# ==================================================================================#
if __name__ == '__main__':
    test_data = MyDatasetFunction(
        dataset_path=r"F:\NPI\train",
        max_token_num=512,
        model_path=r"F:\Python\codet5-base",
    )
    NPI_list = []
    for data in tqdm(test_data):
        if len(data["labels"]) > 6:
            print(len(data["labels"]))
