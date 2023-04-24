'\nDataset to be used for APPS Training\n'
import torch
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json

class Dataset(torch.utils.data.Dataset):

    def __init__(self, Path_to_dataset='F:/ECG/train', Maximum_number_of_tokens=1024, Model_path='codet5-base'):
        self.Path_to_dataset = Path_to_dataset
        self.Maximum_number_of_tokens = Maximum_number_of_tokens
        self.Tokenized_word_list = RobertaTokenizer.from_pretrained(Model_path)
        self.Total_Data_List = []
        self.Total_Data_Dictionary = {}
        self.Initialization_Functions()

    def Initialization_function(self):
        Total_Data_List = []
        Skipped_Questions_List = []
        List_of_data_sets = os.listdir(f'{self.Path_to_dataset}')
        for question in tqdm(List_of_data_sets):
            with open(f'{self.Path_to_dataset}/{question}/Question.txt', 'r', encoding='UTF-8') as f:
                Total_problem_text = f.read()
            Total_problem_text_tensor_dictionary = self.Tokenized_word_list(Total_problem_text, return_tensors='pt')
            if (len(Total_problem_text_tensor_dictionary['input_ids'][0]) > 1278):
                continue
            with open(f'{self.Path_to_dataset}/{question}/Accepted.txt', 'r', encoding='UTF-8') as f:
                Tag_Codes = f.read()
            Tag_code_tensor_dictionary = self.Tokenized_word_list(Tag_Codes, return_tensors='pt')
            if (len(Tag_code_tensor_dictionary['input_ids'][0]) > 766):
                continue
            List_of_slow_code_sets = os.listdir(f'{self.Path_to_dataset}/{question}/Acc_tle_solutions')
            for certain_slow_code in List_of_slow_code_sets:
                with open(f'{self.Path_to_dataset}/{question}/Acc_tle_solutions/{certain_slow_code}', 'r', encoding='UTF-8') as f:
                    slow_code = f.read()
                slow_code_tensor_dictionary = self.Tokenized_word_list(slow_code, return_tensors='pt')
                if (len(slow_code_tensor_dictionary['input_ids'][0]) > 766):
                    continue
                Input_Features = f'''{Total_problem_text}\n{slow_code}\nEfficient code:\n'''
                Tag_Codes = Tag_Codes.replace('(function', '( function')
                data_tuple = (Input_Features, Tag_Codes, f'{self.Path_to_dataset}/{question}/Acc_tle_solutions/{certain_slow_code}')
                Total_Data_List.append(data_tuple)
        self.Total_Data_List = Total_Data_List
        print(f'[0:36m========================= {len(Total_Data_List)} ')

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, Index):
        (Input_Features, Tag_Codes, Code_path) = self.Total_Data_List[Index]
        Input_Features = Input_Features[:150000]
        Tag_Codes = Tag_Codes[:150000]
        Feature_code_dictionary = self.Tokenized_word_list(Input_Features, padding=True, truncation=True, max_length=self.Maximum_number_of_tokens, return_tensors='pt')
        Feature_tensor = Feature_code_dictionary['input_ids']
        Tag_encoding_dictionary = self.Tokenized_word_list(Tag_Codes, padding=True, truncation=True, max_length=self.Maximum_number_of_tokens, return_tensors='pt')
        Label_tensor = Tag_encoding_dictionary['input_ids']
        return {'input_ids': Feature_tensor[0], 'labels': Label_tensor[0], 'Input_Features': Input_Features, 'Label_Code': Tag_Codes}
if (__name__ == '__main__'):
    Test_data = Dataset(Path_to_dataset='CEO/dev', Maximum_number_of_tokens=1024)
    print(Test_data[0])