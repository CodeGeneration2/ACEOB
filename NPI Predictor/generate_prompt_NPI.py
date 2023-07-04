# -*- coding: utf-8 -*-
import os

import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration

import wandb
wandb.login(key="")
print('\033[0:34m===========================================Come on, go for it!========================================\033[m')

model_path = './trained_model_parameters_codet5-base_NPI-No regression/checkpoint-179880'
# model_path = './CodeBERT-base'


# ------------------------------------ Tokenization vocabulary ------------------------------------#
Tokenization_vocabulary = RobertaTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)


# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
model.to(device)



def generate_NPI(code_text):
    a_test_tensor = Tokenization_vocabulary(code_text, truncation=True, max_length=512, return_tensors="pt")
    # print(a_train_tensor)
    # =================================================== simply generate a single sequence ===========================
    with torch.no_grad():
        generated_Token_list = model.generate(a_test_tensor['input_ids'].to(device))
    NPI = Tokenization_vocabulary.decode(generated_Token_list[0], skip_special_tokens=True)

    return NPI
