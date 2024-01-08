# coding=utf-8

from statistics import mean

import torch

import os

from tqdm import tqdm
from transformers import AutoTokenizer
import random
import json

# ################################################################# MyDataset Class #####################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_token_count=512, model_path='codet5-base', train_or_predict=""):

        self.dataset_path = dataset_path
        self.max_token_count = max_token_count
        self.model_path = model_path
        self.train_or_predict = train_or_predict

        # ------------------------------------ Tokenizer Vocabulary ------------------------------------#
        self.tokenizer_vocab = AutoTokenizer.from_pretrained(self.model_path)

        self.total_data_list = []  # Should be set in initialize_function()
        self.total_data_dict = {}  # Should be set in initialize_function()

        # ========================= Initialize Function (Import data from local) ===============#
        self.initialize_function()

    # =========================================== Initialize Function (Import data from local) =========================================#
    def initialize_function(self):
        """
        Import data from local
        Returns:
            self.total_data_list = total_data_list
            self.total_data_dict = total_data_dict

        Assume self.dataset_root_path is set to folderName/data
        """

        total_data_list = []

        dataset_list = os.listdir(f"{self.dataset_path}")

        for certain_question in tqdm(dataset_list):
            # ============================================================= Features ====================================#
            code_list = os.listdir(f"{self.dataset_path}/{certain_question}")
            for certain_code in code_list:
                with open(f"{self.dataset_path}/{certain_question}/{certain_code}", 'r', encoding='UTF-8') as f:
                    code = f.read()

                input_feature = f"{code}\nEvaluate the code running time:"

                runtime = str(int(certain_code.split("KB,Standard Time ")[-1].split(" ms,")[0]))

                # ---------------------------------------- Data unit ----------------------------------------#
                certain_data_tuple = (input_feature, runtime, f"{self.dataset_path}/{certain_question}/{certain_code}")
                # -------------------------- Add to total data list -----------#
                total_data_list.append(certain_data_tuple)

        # ----------------------- Randomize list for comparison -------------#
        random.shuffle(total_data_list)

        self.total_data_list = total_data_list

        print(f'\033[0:36m========================== Loaded {len(total_data_list)} Training Data ==================\033[m')

    def __len__(self):
        return len(self.total_data_list)

    # ================================================ Iteration Function ==================================#
    def __getitem__(self, index):
        input_feature, runtime, code_path = self.total_data_list[index]

        # ---------------------------- Never delete. Delaying bug fixes. --------------#
        input_feature = input_feature[:150000]
        runtime = runtime[:150000]

        # ------------------------------------- Encoding ---------------------------------------------#
        feature_encoding_dict = self.tokenizer_vocab(input_feature, padding='max_length', truncation=True, max_length=self.max_token_count, return_tensors="pt")

        # ------------------------------------- Encoding ---------------------------------------------#
        label_encoding_dict = self.tokenizer_vocab(runtime, padding='max_length', max_length=6, return_tensors="pt")

        if self.train_or_predict == "train":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoding_dict["input_ids"] = feature_encoding_dict['input_ids'].squeeze()
            feature_encoding_dict["attention_mask"] = feature_encoding_dict['attention_mask'].squeeze()
            feature_encoding_dict["labels"] = label_encoding_dict['input_ids'].squeeze()

        elif self.train_or_predict == "predict":
            # ------------------------------------- Encoding ---------------------------------------------#
            feature_encoding_dict["input_feature"] = input_feature
            feature_encoding_dict["runtime"] = runtime
            feature_encoding_dict["code_path"] = code_path

        else:
            print("Error self.train_or_predict")

        if "incoder" in self.model_path:
            feature_encoding_dict.pop("token_type_ids", None)

        return feature_encoding_dict



# ==================================================================================#
if __name__ == '__main__':
    test_data = MyDataset(
        dataset_path=r"F:\0Work3: Expert Group - Efficient Code Generation - Paper - Four Times Rejected\5 - Fifth Submission - Split Post - Pure Expert Group\Dataset\7- Runtime Training Set - 5 and 6 - AST Combined Version\train",
        max_token_count=512,
        model_path=r"F:\PureData\LargeLanguageModelParameters\codet5-base",
        train_or_predict="train"
    )
    NPI_list = []
    for data in tqdm(test_data):
        if len(data["labels"]) > 6:
            print(len(data["labels"]))
