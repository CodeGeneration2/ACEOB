# -*- coding: utf-8 -*-
import json
import os
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration

# ######################################################################################################################
model_path = './CodeT5_Pre_Time'
code_path = "../../ACEOB-NPI/test"
predict_mode = 'time'        # 'time' or 'NPI'


# ######################################################################################################################
# ------------------------------------ Tokenizer Vocabulary ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
model.to(device)


def model_predict_function(code_text):
    test_tensor = tokenizer(code_text, truncation=True, max_length=512, return_tensors="pt")
    # print(test_tensor)
    # =================================================== simply generate a single sequence ===========================
    with torch.no_grad():
        generated_token_list = model.generate(test_tensor['input_ids'].to(device))

    model_prediction = tokenizer.decode(generated_token_list[0], skip_special_tokens=True)

    return model_prediction


# #######################################################################################################
def overall_evaluation_function_predict_time(evaluation_set_path):
    error_dict = {'predicted_time': [], 'label_time': []}

    code_list = os.listdir(f"{evaluation_set_path}")
    for code_file in tqdm(code_list):
        # ---------------------------- Save record file -----------------------------------------------------------#
        with open(f"{evaluation_set_path}/{code_file}", 'r', encoding='UTF-8') as f:
            code_text = f.read().strip()

        predicted_time = model_predict_function(code_text)
        label_time = int(code_file.split(",Running time ")[-1].split(" ms,")[0])

        error_dict['predicted_time'].append(predicted_time)
        error_dict['label_time'].append(label_time)

    return error_dict


# #######################################################################################################
def overall_evaluation_function_predict_NPI(evaluation_set_path):
    error_dict = {'predicted_NPI': [], 'label_NPI': []}

    code_list = os.listdir(f"{evaluation_set_path}")
    for code_file in tqdm(code_list):
        # ---------------------------- Save record file -----------------------------------------------------------#
        with open(f"{evaluation_set_path}/{code_file}", 'r', encoding='UTF-8') as f:
            code_text = f.read().strip()

        predicted_NPI = model_predict_function(code_text)
        label_NPI = float(code_file.split(" ms,NPI ")[-1].split(",")[0])

        error_dict['predicted_NPI'].append(predicted_NPI)
        error_dict['label_NPI'].append(label_NPI)

    return error_dict


# ############################################################ Entry Point ####################################################
if __name__ == '__main__':
    if predict_mode == 'NPI':
        error_dict = overall_evaluation_function_predict_NPI(evaluation_set_path=f"{code_path}")
    elif predict_mode == 'time':
        error_dict = overall_evaluation_function_predict_time(evaluation_set_path=f"{code_path}")


    # Save dictionary to file
    with open('error_dict.json', 'w') as f:
        json.dump(error_dict, f)
