
import os
from numpy import mean

import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from model import Modle
from train_Dataset import TrainDatasetClass
from test_Dataset import Test_set_Dataset_class
from sacrebleu.metrics import BLEU, CHRF, TER


Tokenized = GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")
Tokenized_Bert_word_list = BertTokenizer.from_pretrained("Bert_Token/", pad_token="[PAD]")

def setup_train_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--topk', default=1, type=int, required=False)
    parser.add_argument('--topp', default=0.5, type=float, required=False)  

    parser.add_argument('--device', default='0', type=str, required=False)  
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--arch', default="EleutherAI/gpt-neo-125M")

    parser.add_argument('--log_path', default='log/Prediction_Log.txt', type=str, required=False)

    parser.add_argument('--epochs', default=1, type=int, required=False)
    parser.add_argument('--batch_size', default=1, type=int, required=False)  
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    parser.add_argument('--Whether_to_use_trained_local_mods', default=1, type=str, required=False)

    parser.add_argument('--seed', type=int, default=666)  
    parser.add_argument('--num_workers', type=int, default=5)

    args = parser.parse_args()

    return args


def set_random_seed_function(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_log_file_function(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    time_Format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(time_Format)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(time_Format)
    logger.addHandler(console)

    return logger


def create_model(args):

    model = Modle(args)


    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1

    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        TopK_Return_Value = torch.topk(logits, top_k)

        TopK_last_token_probability = TopK_Return_Value[0][..., -1, None]

        index_removes_Boolean_matrix = logits < TopK_last_token_probability

        logits[index_removes_Boolean_matrix] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        Cumulative_probability = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        filtering_index_removes_Boolean_matrix = Cumulative_probability > top_p

        filtering_index_removes_Boolean_matrix[..., 1:] = filtering_index_removes_Boolean_matrix[..., :-1].clone()
        filtering_index_removes_Boolean_matrix[..., 0] = 0

        index_remove_id_matrix = sorted_indices[filtering_index_removes_Boolean_matrix]

        logits[index_remove_id_matrix] = filter_value

    return logits


def complementary_fill_function(A_batch_data):
    title_Dictionary = Tokenized_Bert_word_list(A_batch_data[0][0], max_length=512, truncation=True, padding=True, return_tensors="pt")
    problem_Description_Subject_Dictionary = Tokenized_Bert_word_list(A_batch_data[0][1], max_length=512, truncation=True, padding=True, return_tensors="pt")
    Input_Description_Dictionary = Tokenized_Bert_word_list(A_batch_data[0][2], max_length=512, truncation=True, padding=True, return_tensors="pt")
    Output_Description_Dictionary = Tokenized_Bert_word_list(A_batch_data[0][3], max_length=512, truncation=True, padding=True, return_tensors="pt")
    Input_output_sample_tests_and_Note_description_dictionary = Tokenized_Bert_word_list(A_batch_data[0][4], max_length=512, truncation=True, padding=True, return_tensors="pt")
    certain_slow_code_dictionary = Tokenized(A_batch_data[0][5], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return [title_Dictionary,problem_Description_Subject_Dictionary,Input_Description_Dictionary,Output_Description_Dictionary,Input_output_sample_tests_and_Note_description_dictionary, certain_slow_code_dictionary], A_batch_data[0][6], A_batch_data[0][7]


def Prediction_generation_function(model, devices, Dataset, Multiple_GPUs, args, Training_set_or_test_set):

    dataloader = DataLoader(dataset=Dataset,  
                         batch_size=args.batch_size,  
                         shuffle=False,  
                         num_workers=args.num_workers,  
                         collate_fn=complementary_fill_function
                         )

    model.eval()



    with torch.no_grad():
        total_BLEU_scores_list = []

        for Batch_index, (Feature_List, label_code, Data_original_path) in enumerate(dataloader):

            for index, A_certain_dictionary in enumerate(Feature_List):
                Feature_List[index]["input_ids"] = A_certain_dictionary["input_ids"].to(devices)
                Feature_List[index]["attention_mask"] = A_certain_dictionary["attention_mask"].to(devices)
                try:
                    Feature_List[index]["token_type_ids"] = A_certain_dictionary["token_type_ids"].to(devices)
                except:
                    pass

            Generated_code = tensor([[102]])
            Generated_code_attention_matrix = tensor([[1]])

            Feature_List.append({"input_ids":Generated_code.to(devices),"attention_mask":Generated_code_attention_matrix.to(devices)})

            Generated_list = []

            for _ in range(len(Feature_List[5]["input_ids"][0])):

                model_output = model(Feature_List).logits

                Mod_prediction_probability_next_Token = model_output[0,-1, :]

                for id in set(Generated_list):  
                    Mod_prediction_probability_next_Token[id] /= 1

                Mod_prediction_probability_next_Token = Mod_prediction_probability_next_Token / 1

                Mod_prediction_probability_next_Token[102] = -float('Inf')

                filtering_logits = top_k_top_p_filtering(Mod_prediction_probability_next_Token, top_k=args.topk, top_p=args.topp)

                Predictive_Token = torch.multinomial(F.softmax(filtering_logits, dim=-1), num_samples=1)

                if Predictive_Token == 50256:
                    break

                Generated_list.append(Predictive_Token.item())

                Feature_List[-1]["input_ids"] = torch.cat((Feature_List[-1]["input_ids"], tensor([[Predictive_Token]]).to(devices)), dim=1)
                Feature_List[-1]["attention_mask"] = torch.cat((Feature_List[-1]["attention_mask"], tensor([[1]]).to(devices)), dim=1)

            Output_Text = Tokenized.batch_decode(Feature_List[-1]["input_ids"])[0].replace("[CLS]","")

            Standard_answer_list = [
                [label_code],
            ]
            model_generation_list = [Output_Text]
            bleu = BLEU()
            bleu_score = bleu.corpus_score(model_generation_list,Standard_answer_list).score
            total_BLEU_scores_list.append(bleu_score)

            with open(f"Generated_code/{Training_set_or_test_set}/{Batch_index},BLEU_score,{bleu_score:.3f}.txt", 'w', encoding='UTF-8') as f:
                f.write(Output_Text)
            with open(f"Generated_code/{Training_set_or_test_set}/{Batch_index},Standard_answer.txt", 'w', encoding='UTF-8') as f:
                f.write(label_code)

            logger.info(f'\033[0:32m{Training_set_or_test_set}, {Batch_index} ,(,{len(dataloader)},),,{bleu_score:.3f},,{len(Feature_List[-2]["input_ids"][0])}\033[m')


    Average_BLEU_score = mean(total_BLEU_scores_list)
    with open(f"Generated_code/{Training_set_or_test_set}/0,BLEU_score,Average_BLEU_score,{Average_BLEU_score:.3f}.txt", 'w', encoding='UTF-8') as f:
        f.write(str(total_BLEU_scores_list))

    logger.info(f'\033[0:34m======================= {Training_set_or_test_set}BLEU_score：{Average_BLEU_score:.3f}。 {Training_set_or_test_set} ========\033[m')

    return Average_BLEU_score


def main():
    args = setup_train_args()


    global logger
    logger = create_log_file_function(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    devices = 'cuda' if args.cuda else 'cpu'
    logger.info('using devices:{}'.format(devices))
    if args.seed:
        set_random_seed_function(args)


    if not os.path.exists(args.Generated_models):
        os.mkdir(args.Generated_models)

    if not os.path.exists("Generated_code/test_code"):
        os.mkdir("Generated_code/test_code")


    model = create_model(args)

    model.to(devices)

    Multiple_GPUs = False
    if args.cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])

    Total_number_parameters = 0
    model_parameter_list = model.parameters()
    for a_layer_parameters in model_parameter_list:
        Total_number_parameters += a_layer_parameters.numel()
    logger.info(f'=========== number of model model_parameter_list: {Total_number_parameters} =================')

    logger.info("=================== loadtrain_data =======================")

    test_data = Test_set_Dataset_class()

    Test_set_BLEU_scores = Prediction_generation_function(model, devices, test_data, Multiple_GPUs, args,"test_code")

    logger.info(f'\033[0:34m======= Test_set_BLEU_scores：{Test_set_BLEU_scores} ===============\033[m')

if __name__ == '__main__':
    main()

class Predictive_code_generation_class():
    def entrance(self):
        main()
