# -*- coding: utf-8 -*-
# print(f":\n{}")
# ############################################################################################################################################

import os
from statistics import mean

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration

print('\033[0:34m===========================================Go for it, all in!========================================\033[m')


# ############################################################################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = './checkpoint-189160'


generated_code_path = "../../Generated_Code"




# ############################################################################################################################################
# ------------------------------------ Tokenize Dictionary ------------------------------------#
tokenize_dictionary = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
model.to(device)


# ############################################################################################################################################
def generate_runtime(code_text):
    code_text = code_text.strip()
    some_test_tensor = tokenize_dictionary(code_text, truncation=True, max_length=512, return_tensors="pt")
    # print(some_training_tensor)
    # =================================================== simply generate a single sequence ===========================
    with torch.no_grad():
        generated_token_list = model.generate(some_test_tensor['input_ids'].to(device))

    runtime = tokenize_dictionary.decode(generated_token_list[0], skip_special_tokens=True)

    return runtime


# #######################################################################################################
def total_evaluation_function(eval_set_path):

    question_list = os.listdir(eval_set_path)
    for question_index in tqdm(question_list):

        various_code_pairs_list = os.listdir(f"{eval_set_path}/{question_index}")
        for some_code_set in various_code_pairs_list:

            if some_code_set.startswith("Efficient-Inefficient Code Pair"):

                code_list = os.listdir(f"{eval_set_path}/{question_index}/{some_code_set}")
                for some_code in code_list:
    
                    if "Predicted Time" in some_code:
                        continue

                    # ---------------------------- Save Record File -----------------------------------------------------------#
                    with open(f"{eval_set_path}/{question_index}/{some_code_set}/{some_code}", 'r', encoding='UTF-8') as f:
                        code_text = f.read().strip()

                    runtime = generate_runtime(code_text)
                    runtime = runtime.replace("/","").replace("\\","")

                    new_name = some_code.replace(".txt", f",Predicted Time {runtime} hrs.txt")
                    # print(runtime)

                    os.rename(f"{eval_set_path}/{question_index}/{some_code_set}/{some_code}", f"{eval_set_path}/{question_index}/{some_code_set}/{new_name}")




# ############################################################ Entry Point ####################################################
if __name__ == '__main__':

    generated_code_list = os.listdir(generated_code_path)
    for some_generated_code_set in generated_code_list:
        total_evaluation_function(eval_set_path=f"{generated_code_path}/{some_generated_code_set}")



