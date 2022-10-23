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
from Model import Model
from train_Dataset import Train_Dataset
from test_Dataset import Test_Dataset
from Prediction_generation_code import main as prediction_generation_code_main
Logger1 = None
Tokenized_GPT_word_list = transformers.GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')

def Receive_command_line_arguments_function():
    Command_line_parser = argparse.ArgumentParser()
    Command_line_parser.add_argument('--device', default='0', type=str, required=False)
    Command_line_parser.add_argument('--cuda', default=1, required=False, help='0:NO,1:Yes')
    Command_line_parser.add_argument('--log_path', default='Log/train_log.txt', type=str, required=False)
    Command_line_parser.add_argument('--epochs', default=4, type=int, required=False)
    Command_line_parser.add_argument('--batch_size', default=1, type=int, required=False)
    Command_line_parser.add_argument('--gradient_accumulation', default=32, type=int, required=False)
    Command_line_parser.add_argument('--lr', default=0.0015, type=float, required=False)
    Command_line_parser.add_argument('--log_step', default=100, type=int, required=False)
    Command_line_parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    Command_line_parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    Command_line_parser.add_argument('--Whether_to_use_trained_local_models', default=0, type=int, required=False, help='0: NO, 1: Yes')
    Command_line_parser.add_argument('--seed', type=int, default=666)
    Command_line_parser.add_argument('--num_workers', type=int, default=5)
    Command_line_parameters = Command_line_parser.parse_args()
    return Command_line_parameters

def Set_random_seed_function(Command_line_parameters):
    torch.manual_seed(Command_line_parameters.seed)
    random.seed(Command_line_parameters.seed)
    np.random.seed(Command_line_parameters.seed)
    if Command_line_parameters.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def Create_log_file_function(Command_line_parameters):
    Logger2 = logging.getLogger(__name__)
    Logger2.setLevel(logging.INFO)
    Time_Format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    Log_File_Writer = logging.FileHandler(filename=Command_line_parameters.log_path)
    Log_File_Writer.setFormatter(Time_Format)
    Log_File_Writer.setLevel(logging.INFO)
    Logger2.addHandler(Log_File_Writer)
    Console = logging.StreamHandler()
    Console.setLevel(logging.DEBUG)
    Console.setFormatter(Time_Format)
    Logger2.addHandler(Console)
    return Logger2

def Complementary_fill_function(Batch_of_data):
    Code = []
    Run_time = []
    Total_runtime = []
    for Certain_data in Batch_of_data:
        Code.append(Certain_data[0])
        Run_time.append(Certain_data[1])
    Code_tensor_dictionary = Tokenized_GPT_word_list(Code, max_length=768, truncation=True, padding=True, return_tensors='pt')
    Total_runtime.append(Run_time)
    Run_time_tensor = torch.Tensor(Total_runtime)
    return [Code_tensor_dictionary, Run_time_tensor]

def train_function(Models, Devices, Training_Data, Test_data, Multiple_GPUs, Command_line_parameters):
    Training_data_loader = DataLoader(dataset=Training_Data, batch_size=Command_line_parameters.batch_size, shuffle=True, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    Models.train()
    Total_number_of_steps = int((((Training_data_loader.__len__() * Command_line_parameters.epochs) / Command_line_parameters.batch_size) / Command_line_parameters.gradient_accumulation))
    Logger1.info(f'================================ Total number of training steps: = {Total_number_of_steps} =========================')
    Optimizer = transformers.AdamW(Models.parameters(), lr=Command_line_parameters.lr, correct_bias=True)
    Loss_function = torch.nn.MSELoss()
    Logger1.info('################################################## starting training ###############################')
    Cumulative_loss_per_gradient = 0
    for certain_round in range(0, Command_line_parameters.epochs):
        Logger1.info(f'========================================= training, epochs: {(certain_round + 1)} =====================')
        Number_of_GPU_memory_runs_out = 0
        Training_loss_list = []
        for (Batch_index, [Code_tensor_dictionary, Run_time_tensor]) in enumerate(Training_data_loader):
            for (Index, certain_tensor_dictionary) in enumerate(Training_data_loader):
                Code_tensor_dictionary['input_ids'] = Code_tensor_dictionary['input_ids'].to(Devices)
                Code_tensor_dictionary['attention_mask'] = Code_tensor_dictionary['attention_mask'].to(Devices)
                Run_time_tensor = Run_time_tensor.to(Devices)
                try:
                    Code_tensor_dictionary['token_type_ids'] = Code_tensor_dictionary['token_type_ids'].to(Devices)
                except:
                    pass
            try:
                Model_Output = Models(Code_tensor_dictionary)
                Losses = Loss_function(Model_Output, Run_time_tensor)
                Training_loss_list.append(Losses.item())
                if ((Batch_index % Command_line_parameters.log_step) == 0):
                    Logger1.info(f'& train, epochs: {(certain_round + 1)} , (Total:{Command_line_parameters.epochs}), {(Batch_index + 1)} (Total:{len(Training_data_loader)}), loss:{Losses:.3f}')
                if Multiple_GPUs:
                    Losses = Losses.mean()
                if (Command_line_parameters.gradient_accumulation > 1):
                    Losses = (Losses / Command_line_parameters.gradient_accumulation)
                Losses.backward()
                torch.nn.utils.clip_grad_norm_(Models.parameters(), Command_line_parameters.max_grad_norm)
                if (((Batch_index + 1) % Command_line_parameters.gradient_accumulation) == 0):
                    Cumulative_loss_per_gradient += Losses.item()
                    Optimizer.step()
                    Optimizer.zero_grad()
            except RuntimeError as exception:
                if ('out of memory' in str(exception)):
                    Number_of_GPU_memory_runs_out += 1
                    Logger1.info(f'#======== WARNING: ran out of memory,times: {Number_of_GPU_memory_runs_out} ========#')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    Logger1.info(str(exception))
                    raise exception
        torch.save(Models.MLP_layer, f'{Command_line_parameters.Generated_models}/MLP_layer.pkl')
        Models.Code_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Code_layer/')
        Logger1.info(f' train: epochs: {(certain_round + 1)} , loss: {mean(Training_loss_list)}')
        Test_average_loss = Prediction_functions(Models, Devices, Test_data, Multiple_GPUs, Command_line_parameters)
    Logger1.info(f'################## train: epochs: {(certain_round + 1)}. End of round. Average loss from training: {mean(Training_loss_list)}. Test average loss: {Test_average_loss} ###################')
    Logger1.info('##################################################################### End training ##################')

def Prediction_functions(Models, Devices, Test_set_data, Multiple_GPUs, Command_line_parameters):
    Logger1.info('=========================== starting Calculating predicted losses =========================')
    Models.eval()
    Loss_function = torch.nn.MSELoss()
    Test_data_loader = DataLoader(Test_set_data, batch_size=Command_line_parameters.batch_size, shuffle=True, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    with torch.no_grad():
        Loss_list = []
        for (Batch_index, [Code_tensor_dictionary, Run_time_tensor]) in enumerate(Test_data_loader):
            Code_tensor_dictionary['input_ids'] = Code_tensor_dictionary['input_ids'].to(Devices)
            Code_tensor_dictionary['attention_mask'] = Code_tensor_dictionary['attention_mask'].to(Devices)
            Run_time_tensor = Run_time_tensor.to(Devices)
            try:
                Code_tensor_dictionary['token_type_ids'] = Code_tensor_dictionary['token_type_ids'].to(Devices)
            except:
                pass
            try:
                Model_Output = Models(Code_tensor_dictionary)
            except:
                continue
            Losses = Loss_function(Model_Output, Run_time_tensor)
            Loss_list.append(Losses.item())
        Logger1.info(f' evaluate test set loss:  {mean(Loss_list):.3f}')
        return mean(Loss_list)

def main():
    Command_line_parameters = Receive_command_line_arguments_function()
    global Logger1
    Logger1 = Create_log_file_function(Command_line_parameters)
    os.environ['CUDA_VISIBLE_DEVICES'] = Command_line_parameters.device
    Devices = ('cuda' if Command_line_parameters.cuda else 'cpu')
    Logger1.info('using :{}'.format(Devices))
    if Command_line_parameters.seed:
        Set_random_seed_function(Command_line_parameters)
    if (not os.path.exists(Command_line_parameters.log_path)):
        os.mkdir(Command_line_parameters.log_path)
    if (not os.path.exists(Command_line_parameters.Generated_models)):
        os.mkdir(Command_line_parameters.Generated_models)
    Models = Model(Command_line_parameters)
    Models.to(Devices)
    Multiple_GPUs = False
    Total_number_of_parameters = 0
    List_of_model_parameters = Models.parameters()
    for (layer, Parameters_of_a_certain_layer) in enumerate(List_of_model_parameters):
        Total_number_of_parameters += Parameters_of_a_certain_layer.numel()
    Logger1.info(f'==============================Total model parameters: {Total_number_of_parameters} ===============================')
    Logger1.info('======================Load training data.... ==========================')
    Training_Data = Train_Dataset()
    Test_data = Test_Dataset()
    train_function(Models, Devices, Training_Data, Test_data, Multiple_GPUs, Command_line_parameters)
if (__name__ == '__main__'):
    main()