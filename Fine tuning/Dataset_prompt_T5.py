# coding=utf-8
"""
Dataset to be used for APPS Training
Note: Success GPT
"""

import torch

import os

from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import AutoTokenizer

import json

# ################################################################# MyDatasetFunction #####################################
class MyDatasetFunction(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_tokens=512, model_path=r'F:\PythonPureData\LargeLanguageModelParameters\codet5-small', include_path=False):

        self.dataset_path = dataset_path
        self.max_tokens = max_tokens
        self.include_path = include_path

        # ------------------------------------ Vocabulary ------------------------------------#
        if "codet5" in model_path:
            # ------------------------------------ Tokenizer vocabulary ------------------------------------#
            self.tokenizer_vocab = RobertaTokenizer.from_pretrained(model_path)
        elif "codegen" in model_path:
            # ------------------------------------ Tokenizer vocabulary ------------------------------------#
            self.tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer_vocab.pad_token = self.tokenizer_vocab.eos_token

        self.total_data_list = []  # Should be set in initialization_function()
        self.total_data_dict = {}  # Should be set in initialization_function()

        # ========================= Initialization function (Import data from local) ===============#
        self.initialization_function()

    # =========================================== Initialization function (Import data from local) =========================================#
    def initialization_function(self):
        """
        Import data from local
        Return:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_root_path is set to folderName/data
        """

        total_data_list = []
        skip_question_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")
        for each_problem in tqdm(dataset_list):
            # ============================================================= Feature ====================================#
            problem_content_list = os.listdir(f"{self.dataset_path}/{each_problem}")
            for each_content in problem_content_list:
                if "efficient-inefficient code pair" in each_content:
                    # ============================================================= Feature ====================================#
                    efficient_inefficient_code_list = os.listdir(f"{self.dataset_path}/{each_problem}/{each_content}")
                    # ============================================== Import each feature set ===================================#
                    for each_code in efficient_inefficient_code_list:
                        if "efficient" in each_code:
                            with open(f"{self.dataset_path}/{each_problem}/{each_content}/{each_code}", 'r', encoding='UTF-8') as f:
                                fast_code = f.read()
                        elif "inefficient" in each_code:
                            with open(f"{self.dataset_path}/{each_problem}/{each_content}/{each_code}", 'r', encoding='UTF-8') as f:
                                slow_code = f.read()

                    input_feature = f"Please optimize the following inefficient code into a more efficient version, while keeping the functionality unchanged:\n{slow_code}\nMore efficient version:\n"
                    each_data_tuple = (input_feature, fast_code, f"{self.dataset_path}/{each_problem}/{each_content}")
                    total_data_list.append(each_data_tuple)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== {len(total_data_list)} training set data loaded ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iteration traversal function ==================================#
    def __getitem__(self, index):
        input_feature, label_code, code_path = self.total_data_list[index]

        # ---------------------------- Will never be deleted. Postpone bug fix. --------------#
        input_feature = input_feature[:150000]
        label_code = label_code[:150000]

        # ------------------------------------- Encoding ---------------------------------------------#
        feature_encoding_dict = self.tokenizer_vocab(input_feature, padding='max_length', truncation=True, max_length=self.max_tokens,  return_tensors="pt")

        # ------------------------------------- Encoding ---------------------------------------------#
        label_encoding_dict = self.tokenizer_vocab(label_code, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")

        if self.include_path:
            feature_encoding_dict = self.tokenizer_vocab(input_feature, return_tensors="pt")
            feature_encoding_dict["input_feature"] = input_feature
            feature_encoding_dict["label_code"] = label_code
            feature_encoding_dict["code_path"] = code_path
        else:
            feature_encoding_dict = self.tokenizer_vocab(input_feature, padding='max_length', truncation=True,
                                            max_length=self.max_tokens, return_tensors="pt")
            label_encoding_dict = self.tokenizer_vocab(label_code, padding='max_length', truncation=True,
                                            max_length=self.max_tokens, return_tensors="pt")
            feature_encoding_dict["input_ids"] = feature_encoding_dict['input_ids'][0]
            feature_encoding_dict["attention_mask"] = feature_encoding_dict['attention_mask'][0]
            feature_encoding_dict["labels"] = label_encoding_dict['input_ids'][0]

        return feature_encoding_dict


# ==================================================================================#
# ==================================================================================#
# ==================================================================================#
if __name__ == '__main__':
    test_data = MyDatasetFunction(
        dataset_path=r"F:\ACEO-Current-Dataset-Paper1\Dataset\46-ACEO\train",
        max_tokens=512
    )
    data = test_data[0]
    print(test_data[0])
