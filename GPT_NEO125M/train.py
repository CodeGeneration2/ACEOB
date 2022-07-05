
import os

from numpy import mean

import transformers
import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
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

from model import Modle
from train_Dataset import TrainDatasetClass
from test_Dataset import Test_set_Dataset_class
from Predictive_generation_code import Predictive_code_generation_class

tokenVocabulary = transformers.GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")

def get_args():

    parser = argparse.ArgumentParser()


    parser.add_argument('--device', default='0', type=str, required=False)  
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--arch', default="EleutherAI/gpt-neo-125M")


    parser.add_argument('--log_path', default='log/TrainLog.txt', type=str, required=False)

    parser.add_argument('--epochs', default=30, type=int, required=False)
    parser.add_argument('--batch_size', default=1, type=int, required=False)  
    parser.add_argument('--gradient_accumulation', default=32, type=int, required=False)

    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=1000, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    parser.add_argument('--Whether_to_use_trained_local_mods', default=0, type=str, required=False)

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


def complementary_fill_function(A_batch_data):
    Question_text = []
    certain_slow_code = []
    label_code = []

    for a_data in A_batch_data:
        Question_text.append(a_data[0])
        certain_slow_code.append(a_data[1])
        label_code.append(a_data[2])

    Question_text_dictionary = tokenVocabulary(Question_text, max_length=512, truncation=True, padding=True, return_tensors="pt")
    certain_slow_code_dictionary = tokenVocabulary(certain_slow_code, max_length=768, truncation=True, padding=True, return_tensors="pt")
    label_code_dictionary = tokenVocabulary(label_code, max_length=766, truncation=True, padding=True, return_tensors="pt")

    batch_size = len(label_code_dictionary["input_ids"])

    Starting_Token_List = [[102]] * batch_size
    Starting_Token = tensor(Starting_Token_List)

    End_Token_List = [[50256]] * batch_size
    End_Token = tensor(End_Token_List)

    label_code_dictionary["input_ids"] = torch.cat((Starting_Token, label_code_dictionary["input_ids"], End_Token), dim=1)

    Attention_Token_List = [[1]] * batch_size
    Attention_Token = tensor(Attention_Token_List)

    label_code_dictionary["attention_mask"] = torch.cat((Attention_Token, label_code_dictionary["attention_mask"], Attention_Token), dim=1)

    New_problem_text_tensor = torch.cat((Question_text_dictionary["input_ids"], certain_slow_code_dictionary["input_ids"], label_code_dictionary["input_ids"]), dim=1)
    New_Problem_Attention_Tensor = torch.cat((Question_text_dictionary["attention_mask"], certain_slow_code_dictionary["attention_mask"], label_code_dictionary["attention_mask"]), dim=1)

    New_problem_text_tensor_dictionary = {"input_ids":New_problem_text_tensor,"attention_mask":New_Problem_Attention_Tensor}


    New_Label_Tensor_List = [[]]
    New_Label_Tensor_List[0].extend([-100] * len(Question_text_dictionary["input_ids"][0]))
    New_Label_Tensor_List[0].extend([-100] * len(certain_slow_code_dictionary["input_ids"][0]))

    New_Label_Tensor = torch.LongTensor(New_Label_Tensor_List)
    New_Label_Tensor = torch.cat((New_Label_Tensor, label_code_dictionary["input_ids"]), dim=1)

    New_Label_Dictionary = {"input_ids":New_Label_Tensor,"attention_mask":New_Label_Tensor}


    return [New_problem_text_tensor_dictionary, certain_slow_code_dictionary, New_Label_Dictionary]


def train_Functions(model, devices, Training_set_data, test_set_data, Multiple_GPUs, args):

    train_dataloader = DataLoader(dataset=Training_set_data,  
                         batch_size=args.batch_size,  
                         shuffle=True,  
                         num_workers=args.num_workers,  
                         collate_fn=complementary_fill_function
                         )

    model.train()

    Total_number_steps = int(train_dataloader.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)

    logger.info(f'-------- Total_number_steps：total training steps = {Total_number_steps} ---------')

    Optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    Losses_accumulated_per_gradient = 0


    for epoch in range(args.epochs):

        start_time = datetime.now()

        GPU_memory_shortage_times = 0

        Training_loss_list = []
        for Batch_index, Feature_List in enumerate(train_dataloader):


            for index, A_certain_dictionary in enumerate(Feature_List):
                Feature_List[index]["input_ids"] = A_certain_dictionary["input_ids"].to(devices)
                Feature_List[index]["attention_mask"] = A_certain_dictionary["attention_mask"].to(devices)
                try:
                    Feature_List[index]["token_type_ids"] = A_certain_dictionary["token_type_ids"].to(devices)
                except:
                    pass

            try:    
                model_output = model(Feature_List)
                loss = model_output.loss

                Training_loss_list.append(loss.item())

                if Batch_index % args.log_step == 0:
                    logger.info(f'\033[0:32m {epoch+1} ,{args.epochs}, {Batch_index+1} ,{len(train_dataloader)},loss： {loss:.3f}\033[m')

                if Multiple_GPUs:
                    loss = loss.mean()

                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (Batch_index + 1) % args.gradient_accumulation == 0:
                    Losses_accumulated_per_gradient += loss.item()
                    Optimizer.step()
                    Optimizer.zero_grad()

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    GPU_memory_shortage_times += 1
                    logger.info(f"#======== WARNING: GPU_memory_shortage_times：ran out of memory,times: {GPU_memory_shortage_times} ========#")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception

        Ending_time = datetime.now()

        model.Final_output_layer.save_pretrained(f"./{args.Generated_models}/Final_output_layer/")

        logger.info(f'\033[0:34m===== , {epoch+1} ,loss：{mean(Training_loss_list)},Running_time：{Ending_time - start_time} =======\033[m')

        test_average_loss = predictive_Functions(model, devices, test_set_data, Multiple_GPUs, args)




def predictive_Functions(model, devices, test_set_data, Multiple_GPUs, args):
    logger.info("============================,================================")

    model.eval()

    test_dataloader = DataLoader(test_set_data,  
                         batch_size=args.batch_size,  
                         shuffle=True,  
                         num_workers=args.num_workers,  
                         collate_fn=complementary_fill_function
                         )

    with torch.no_grad():
        Loss_List = []
        for Batch_index, Feature_List in enumerate(test_dataloader):

            for index, A_certain_dictionary in enumerate(Feature_List):
                Feature_List[index]["input_ids"] = A_certain_dictionary["input_ids"].to(devices)
                Feature_List[index]["attention_mask"] = A_certain_dictionary["attention_mask"].to(devices)
                try:
                    Feature_List[index]["token_type_ids"] = A_certain_dictionary["token_type_ids"].to(devices)
                except:
                    pass

            try:
                model_output = model(Feature_List)
            except:
                continue
            loss = model_output.loss

            Loss_List.append(loss.item())

            break

        logger.info(f'\033[0:32m, loss： {mean(Loss_List):.3f}\033[m')

        logger.info('\033[0:34m==========================,==============================\033[m')

        return mean(Loss_List)


def main():
    args = get_args()

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

    logger.info(f'=========== model: {Total_number_parameters} =================')

    logger.info("=================== loadtrain_data=======================")

    train_data = TrainDatasetClass()
    test_data = Test_set_Dataset_class()

    train_Functions(model, devices, train_data,test_data, Multiple_GPUs, args)

    Predictive_code_generation = Predictive_code_generation_class()
    Predictive_code_generation.entrance()

if __name__ == '__main__':
    main()
