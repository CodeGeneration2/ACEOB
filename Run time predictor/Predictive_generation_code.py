
import os
from numpy import mean

import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
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

from model import Model
from predicted_Dataset import Predicted_Dataset_Class

from sacrebleu.metrics import BLEU, CHRF, TER

Tokenized = GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")


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

    model = Model(args)


    return model



def complementary_fill_function(A_batch_data):
    code_dictionary = Tokenized(A_batch_data[0], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return code_dictionary


def Prediction_generation_function(model, devices, Dataset, Multiple_GPUs, args, Training_set_or_test_set):

    dataloader = DataLoader(dataset=Dataset,  
                         batch_size=args.batch_size,  
                         shuffle=False,  
                         num_workers=args.num_workers,  
                         collate_fn=complementary_fill_function
                         )

    model.eval()


    with torch.no_grad():
        Total_list_running_times = []

        for Batch_index, code_dictionary in enumerate(dataloader):

            code_dictionary["input_ids"] = code_dictionary["input_ids"].to(devices)
            code_dictionary["attention_mask"] = code_dictionary["attention_mask"].to(devices)
            try:
                code_dictionary["token_type_ids"] = code_dictionary["token_type_ids"].to(devices)
            except:
                pass

            model_output = model(code_dictionary)

            Predicted_running_time = model_output.cpu().numpy().tolist()
            Predicted_running_time = Predicted_running_time[0][0]

            print(type(Predicted_running_time))

            Total_list_running_times.append(Predicted_running_time)
            logger.info(f'\033[0:34m======================={Batch_index}. model_output：{Predicted_running_time}   ========\033[m')

            break


    Average_running_time = mean(Total_list_running_times)

    with open(f"Forecast time.txt", 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_running_times))

    logger.info(f'\033[0:34m======================= Average_running_time：{Average_running_time}   ======\033[m')

    return Average_running_time


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

    logger.info("=================== loadtrain_data======================")

    forecast_Data = Predicted_Dataset_Class()

    Training_set_BLEU_score = Prediction_generation_function(model, devices, forecast_Data, Multiple_GPUs, args,"Training_set")

    logger.info(f'\033[0:34m======= Training_set_BLEU_score：{Training_set_BLEU_score}  ===============\033[m')

if __name__ == '__main__':
    main()

class Predictive_code_generation_class():
    def entrance(self):
        main()
