'\nDataset to be used for APPS Training\n'
import torch
import os
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json

class Dataset(torch.utils.data.Dataset):

    def __init__(self, Data_set_path='F:/ECG/train', Maximum_number_of_tokens=2048, Model_path='codet5-base', Difficulty='all'):
        self.Total_data_set_paths = Data_set_path
        self.Maximum_Number_of_Tokens = Maximum_number_of_tokens
        self.Difficulty = Difficulty
        self.Tokenized_Word_List = RobertaTokenizer.from_pretrained(Model_path)
        self.Total_Data_List = []
        self.Total_Data_Dictionary = {}
        for set in ['train', 'dev', 'test']:
            self.Data_set_paths = f'{self.Total_data_set_paths}/{set}'
            Data_Lists = self.Initialization_function()
            self.Total_Data_List.extend(Data_Lists)
        print(f'[0:36m==========================  {len(self.Total_Data_List)}===========[m')

    def Initialization_function(self):

        Total_data_list = []
        List_of_data_sets = os.listdir(f'{self.Data_set_paths}')
        for certain_problem in tqdm(List_of_data_sets):
            with open(f'{self.Data_set_paths}/{certain_problem}/Accepted.txt', 'r', encoding='UTF-8') as f:
                Tag_code = f.read()
            with open(f'{self.Data_set_paths}/{certain_problem}/Accepted run time.txt', 'r', encoding='UTF-8') as f:
                Running_time = f.read().split('accepted,')[1].split(' ms,')[0]
            Total_data_list.append((Tag_code, Running_time))
            List_of_slow_codesets = os.listdir(f'{self.Data_set_paths}/{certain_problem}/Acc_tle_solutions')
            for slow_code in List_of_slow_codesets:
                with open(f'{self.Data_set_paths}/{certain_problem}/Acc_tle_solutions/{slow_code}', 'r', encoding='UTF-8') as f:
                    Slow_code = f.read()
                Running_time = slow_code.split(',')[1].split(' ms')[0]
                Total_data_list.append((Slow_code, Running_time))
            List_of_fast_codesets = os.listdir(f'{self.Data_set_paths}/{certain_problem}/Acc_solutions')
            for fast_code in List_of_fast_codesets:
                with open(f'{self.Data_set_paths}/{certain_problem}/Acc_solutions/{fast_code}', 'r', encoding='UTF-8') as f:
                    Fast_Code = f.read()
                Running_time = fast_code.split(',')[1].split(' ms')[0]
                Total_data_list.append((Fast_Code, Running_time))
        return Total_data_list

    def __len__(self):
        return len(self.Total_Data_List)

    def __getitem__(self, Index):
        (Codes, Tag_code) = self.Total_Data_List[Index]
        Input_features = f'''{Codes}
Code run time is:'''
        Feature_coding_dictionary = self.Tokenized_Word_List(Input_features, padding=True, truncation=True, max_length=self.Maximum_Number_of_Tokens, return_tensors='pt')
        Feature_tensor = Feature_coding_dictionary['input_ids']
        Label_encoding_dictionary = self.Tokenized_Word_List(Tag_code, padding=True, truncation=True, max_length=self.Maximum_Number_of_Tokens, return_tensors='pt')
        Label_tensor = Label_encoding_dictionary['input_ids']
        return {'input_ids': Feature_tensor[0], 'labels': Label_tensor[0]}
if (__name__ == '__main__'):
    Test_Data = Dataset(Data_set_paths='./train', Maximum_Number_of_Tokens=1024)
    print(Test_Data[0])