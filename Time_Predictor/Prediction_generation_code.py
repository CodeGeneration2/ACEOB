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
from Model import Model
from train_Dataset import Train_Dataset
from test_Dataset import Test_Dataset
from Prediction_Dataset import Prediction_Dataset
from sacrebleu.metrics import BLEU, CHRF, TER
projection_log = None
Tokenized_GPT = GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')

def Receive_command_line_arguments_function():
    Command_line_parser = argparse.ArgumentParser()
    Command_line_parser.add_argument('--device', default='0', type=str, required=False)
    Command_line_parser.add_argument('--GPT_arch', default='EleutherAI/gpt-neo-125M')
    Command_line_parser.add_argument('--heads', default=48, type=int)
    Command_line_parser.add_argument('--RELU', default=0, type=int, required=False, help='0: NO, 1: Yes')
    Command_line_parser.add_argument('--Maximum_length_pattern_of_generated_code', default=0, type=int, required=False)
    Command_line_parser.add_argument('--Maximum_length_of_generated_code', default=748, type=int, required=False)
    Command_line_parser.add_argument('--topk', default=3, type=int, required=False)
    Command_line_parser.add_argument('--topp', default=0.7, type=float, required=False)
    Command_line_parser.add_argument('--Temperature', default=0.25, type=float, required=False)
    Command_line_parser.add_argument('--Train_set_interval', default=500, type=int, required=False)
    Command_line_parser.add_argument('--Test_set_interval', default=1, type=int, required=False)
    Command_line_parser.add_argument('--cuda', default=1, required=False)
    Command_line_parser.add_argument('--log_path', default='Log/Projection_log.txt', type=str, required=False)
    Command_line_parser.add_argument('--batch_size', default=1, type=int, required=False)
    Command_line_parser.add_argument('--log_step', default=100, type=int, required=False)
    Command_line_parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    Command_line_parser.add_argument('--Whether_to_use_trained_local_models', default=1, type=int, required=False, help='0: NO, 1: Yes')
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
    Prediction_logs = logging.getLogger(__name__)
    Prediction_logs.setLevel(logging.INFO)
    Time_Format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    Log_File_Writer = logging.FileHandler(filename=Command_line_parameters.log_path)
    Log_File_Writer.setFormatter(Time_Format)
    Log_File_Writer.setLevel(logging.INFO)
    Prediction_logs.addHandler(Log_File_Writer)
    Console = logging.StreamHandler()
    Console.setLevel(logging.DEBUG)
    Console.setFormatter(Time_Format)
    Prediction_logs.addHandler(Console)
    return Prediction_logs

def Complementary_fill_function(Batch_of_data):
    Code_tensor_dictionary = Tokenized_GPT(Batch_of_data[0], max_length=768, truncation=True, padding=True, return_tensors='pt')
    return Code_tensor_dictionary

def Prediction_generation_function(Models, Devices, Data_sets, Command_line_parameters):
    Data_Loader = DataLoader(dataset=Data_sets, batch_size=Command_line_parameters.batch_size, shuffle=False, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    Models.eval()
    projection_log.info('############################################## starting Prediction_generation_code ###########')
    with torch.no_grad():
        Total_list_of_running_times = []
        for (Batch_index, Code_tensor_dictionary) in enumerate(Data_Loader):
            Code_tensor_dictionary['input_ids'] = Code_tensor_dictionary['input_ids'].to(Devices)
            Code_tensor_dictionary['attention_mask'] = Code_tensor_dictionary['attention_mask'].to(Devices)
            try:
                Code_tensor_dictionary['token_type_ids'] = Code_tensor_dictionary['token_type_ids'].to(Devices)
            except:
                pass
            Model_Output = Models(Code_tensor_dictionary)
            Predicted_runtime = Model_Output.cpu().numpy().tolist()
            Predicted_runtime = Predicted_runtime[0][0]
            Total_list_of_running_times.append(Predicted_runtime)
            projection_log.info(f'[0:34m======================={Batch_index}. Model output: {Predicted_runtime}   ========[m')
    Average_running_time = mean(Total_list_of_running_times)
    with open(f'Predicted time.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_running_times))
    projection_log.info(f'[0:34m======================= Average Predicted time: {Average_running_time}    =======[m')
    return Average_running_time

def main(Training_Data, Test_data, Number_of_certain_rounds):
    Command_line_parameters = Receive_command_line_arguments_function()
    global projection_log
    projection_log = Create_log_file_function(Command_line_parameters)
    os.environ['CUDA_VISIBLE_DEVICES'] = Command_line_parameters.device
    Devices = ('cuda' if Command_line_parameters.cuda else 'cpu')
    projection_log.info('using:{}'.format(Devices))
    if Command_line_parameters.seed:
        Set_random_seed_function(Command_line_parameters)
    if (not os.path.exists(f'Generated_code/Round_{Number_of_certain_rounds}_prediction_code/Train')):
        os.makedirs(f'Generated_code/Round_{Number_of_certain_rounds}_prediction_code/Train')
    if (not os.path.exists(f'Generated_code/Round_{Number_of_certain_rounds}_prediction_code/Test')):
        os.makedirs(f'Generated_code/Round_{Number_of_certain_rounds}_prediction_code/Test')
    Models = Model(Command_line_parameters)
    Models.to(Devices)
    Multiple_GPUs = False
    Total_number_of_parameters = 0
    List_of_model_parameters = Models.parameters()
    for Parameters_of_a_certain_layer in List_of_model_parameters:
        Total_number_of_parameters += Parameters_of_a_certain_layer.numel()
    projection_log.info(f'================When tested Model Total number of model parameters : {Total_number_of_parameters} ====================')
    Prediction_data = Prediction_Dataset()
    Prediction_time = Prediction_generation_function(Models, Devices, Prediction_data, Command_line_parameters)
    projection_log.info(f'================= Prediction_time: {Prediction_time} ==')
    projection_log.info('######################################## End Prediction #####################')
if (__name__ == '__main__'):
    Training_Data = Train_Dataset()
    Test_data = Test_Dataset()
    main(Training_Data, Test_data, '0')