import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from numpy import mean
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

def Prediction_generation_code_generate_prompt_function(Global_model_path, Data_set_path='generate'):
    Tokenized_word_list = AutoTokenizer.from_pretrained(Global_model_path)
    model = T5ForConditionalGeneration.from_pretrained(Global_model_path)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    if (not os.path.exists(f'generate')):
        os.makedirs(f'generate')
    Total_list_of_CE_scores = []
    List_of_data_sets = os.listdir(f'{Data_set_path}')
    for code in tqdm(List_of_data_sets):
        if (code.split(',')[1] != 'ans.txt'):
            with open(f'{Data_set_path}/{code}', 'r', encoding='UTF-8') as f:
                Codes = f.read()
            Coded_dictionary = Tokenized_word_list(Codes, padding=True, truncation=True, return_tensors='pt')
            Code_tensor = Coded_dictionary['input_ids']
            Generate_Token = model.generate(Code_tensor.to(device), max_length=5, min_length=1, num_beams=5, early_stopping=True)
            Prediction_Codes = Tokenized_word_list.decode(Generate_Token[0], skip_special_tokens=True)
            Total_list_of_CE_scores.append(int(Prediction_Codes))
    with open(f'RES.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_CE_scores))
    print(f'{len(Total_list_of_CE_scores)}')
    print(f'{mean(Total_list_of_CE_scores)} ')
if (__name__ == '__main__'):

    Prediction_generation_code_generate_prompt_function(Global_model_path='codeT5_base/checkpoint-13950', Data_set_paths='./generate')