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
from Model import E_code_Model
from No_expert_Model import No_expert_Model
from GPT_Neo_Model import GPT_Neo_Model
from No_expert_train_Dataset import No_expert_Train_Dataset
from No_expert_test_Dataset import No_expert_Test_Dataset
from train_Dataset import Train_Dataset
from GPT_Neo_train_Dataset import GPT_Neo_Train_Dataset
from No_expert_test_Dataset import No_expert_Test_Dataset
from test_Dataset import Test_Dataset
from GPT_Neo_test_Dataset import GPT_Neo_Test_Dataset
from Prediction_generation_code import main as prediction_generation_code_main
Logger = None
Tokenized_GPT_word_list = transformers.GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')
Tokenized_Bert_word_list = BertTokenizer.from_pretrained('Bert_Tokenizer/', pad_token='[PAD]')

def Receive_command_line_arguments():
    Command_line_parser = argparse.ArgumentParser()
    Command_line_parser.add_argument('--device', default='0', type=str, required=False)
    Command_line_parser.add_argument('--cuda', default=1, required=False, help='0:NO,1:Yes')
    Command_line_parser.add_argument('--task', default=0, type=int, required=False, help='0: E-code, 1: No-expert group, 2: GPT-Neo')
    Command_line_parser.add_argument('--GPT_arch', default='EleutherAI/gpt-neo-125M')
    Command_line_parser.add_argument('--heads', default=48, type=int)
    Command_line_parser.add_argument('--RELU', default=0, type=int, required=False, help='0: NO, 1: Yes')
    Command_line_parser.add_argument('--log_path', default='Log/train_log.txt', type=str, required=False)
    Command_line_parser.add_argument('--epochs', default=15, type=int, required=False)
    Command_line_parser.add_argument('--batch_size', default=1, type=int, required=False)
    Command_line_parser.add_argument('--gradient_accumulation', default=32, type=int, required=False)
    Command_line_parser.add_argument('--lr', default=0.00015, type=float, required=False)
    Command_line_parser.add_argument('--log_step', default=1000, type=int, required=False)
    Command_line_parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    Command_line_parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False)
    Command_line_parser.add_argument('--Whether_to_use_trained_local_models', default=0, type=int, required=False, help='0: NO, 1: Yes')
    Command_line_parser.add_argument('--seed', type=int, default=666)
    Command_line_parser.add_argument('--num_workers', type=int, default=5)
    Command_line_parameters = Command_line_parser.parse_args()
    return Command_line_parameters

def Setting_random_seed_functions(Command_line_parameters):
    torch.manual_seed(Command_line_parameters.seed)
    random.seed(Command_line_parameters.seed)
    np.random.seed(Command_line_parameters.seed)
    if Command_line_parameters.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def Create_log_file_functions(Command_line_parameters):
    Logs = logging.getLogger(__name__)
    Logs.setLevel(logging.INFO)
    Time_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    Log_file_writers = logging.FileHandler(filename=Command_line_parameters.log_path)
    Log_file_writers.setFormatter(Time_format)
    Log_file_writers.setLevel(logging.INFO)
    Logs.addHandler(Log_file_writers)
    Console = logging.StreamHandler()
    Console.setLevel(logging.DEBUG)
    Console.setFormatter(Time_format)
    Logs.addHandler(Console)
    return Logs

def E_code_complementary_fill_function(batch_of_data):
    Title = []
    Body_of_the_problem_description = []
    Input_description = []
    Output_description = []
    Input_Output_Sample_Tests_and_Note_Descriptions = []
    slow_code = []
    The_tag_code = []
    for data in batch_of_data:
        Title.append(data[0])
        Body_of_the_problem_description.append(data[1])
        Input_description.append(data[2])
        Output_description.append(data[3])
        Input_Output_Sample_Tests_and_Note_Descriptions.append(data[4])
        slow_code.append(data[5])
        The_tag_code.append(data[6])
    Title_tensor_dictionary = Tokenized_Bert_word_list(Title, max_length=512, truncation=True, padding=True, return_tensors='pt')
    Question_description_subject_tensor_dictionary = Tokenized_Bert_word_list(Body_of_the_problem_description, max_length=512, truncation=True, padding=True, return_tensors='pt')
    Input_Description_Tensor_Dictionary = Tokenized_Bert_word_list(Input_description, max_length=512, truncation=True, padding=True, return_tensors='pt')
    Output_description_tensor_dictionary = Tokenized_Bert_word_list(Output_description, max_length=512, truncation=True, padding=True, return_tensors='pt')
    Input_and_output_sample_tests_and_Note_description_tensor_dictionary = Tokenized_Bert_word_list(Input_Output_Sample_Tests_and_Note_Descriptions, max_length=512, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(slow_code, max_length=768, truncation=True, padding=True, return_tensors='pt')
    Tag_code_tensor_dictionary = Tokenized_GPT_word_list(The_tag_code, max_length=766, truncation=True, padding=True, return_tensors='pt')
    Batch_size = len(Tag_code_tensor_dictionary['input_ids'])
    Start_Token_List = ([[102]] * Batch_size)
    Start_Token = tensor(Start_Token_List)
    End_Token_List = ([[50256]] * Batch_size)
    End_Token = tensor(End_Token_List)
    Tag_code_tensor_dictionary['input_ids'] = torch.cat((Start_Token, Tag_code_tensor_dictionary['input_ids'], End_Token), dim=1)
    Attention_Token_List = ([[1]] * Batch_size)
    Attention_Token = tensor(Attention_Token_List)
    Tag_code_tensor_dictionary['attention_mask'] = torch.cat((Attention_Token, Tag_code_tensor_dictionary['attention_mask'], Attention_Token), dim=1)
    return [Title_tensor_dictionary, Question_description_subject_tensor_dictionary, Input_Description_Tensor_Dictionary, Output_description_tensor_dictionary, Input_and_output_sample_tests_and_Note_description_tensor_dictionary, slow_code_tensor_dictionary, Tag_code_tensor_dictionary]

def No_expert_completion_fill_function(batch_of_data):
    Total_problem_text = []
    slow_code = []
    The_tag_code = []
    for data in batch_of_data:
        Total_problem_text.append(data[0])
        slow_code.append(data[1])
        The_tag_code.append(data[2])
    Total_Question_Text_Tensor_Dictionary = Tokenized_Bert_word_list(Total_problem_text, max_length=2048, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(slow_code, max_length=768, truncation=True, padding=True, return_tensors='pt')
    Tag_code_tensor_dictionary = Tokenized_GPT_word_list(The_tag_code, max_length=766, truncation=True, padding=True, return_tensors='pt')
    Batch_size = len(Tag_code_tensor_dictionary['input_ids'])
    Start_Token_List = ([[102]] * Batch_size)
    Start_Token = tensor(Start_Token_List)
    End_Token_List = ([[50256]] * Batch_size)
    End_Token = tensor(End_Token_List)
    Tag_code_tensor_dictionary['input_ids'] = torch.cat((Start_Token, Tag_code_tensor_dictionary['input_ids'], End_Token), dim=1)
    Attention_Token_List = ([[1]] * Batch_size)
    Attention_Token = tensor(Attention_Token_List)
    Tag_code_tensor_dictionary['attention_mask'] = torch.cat((Attention_Token, Tag_code_tensor_dictionary['attention_mask'], Attention_Token), dim=1)
    return [Total_Question_Text_Tensor_Dictionary, slow_code_tensor_dictionary, Tag_code_tensor_dictionary]

def GPT_complementary_padding_function(batch_of_data):
    Total_problem_text = []
    slow_code = []
    The_tag_code = []
    for data in batch_of_data:
        Total_problem_text.append(data[0])
        slow_code.append(data[1])
        The_tag_code.append(data[2])
    Total_Question_Text_Tensor_Dictionary = Tokenized_Bert_word_list(Total_problem_text, max_length=512, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(slow_code, max_length=768, truncation=True, padding=True, return_tensors='pt')
    Tag_code_tensor_dictionary = Tokenized_GPT_word_list(The_tag_code, max_length=766, truncation=True, padding=True, return_tensors='pt')
    Batch_size = len(Tag_code_tensor_dictionary['input_ids'])
    Start_Token_List = ([[102]] * Batch_size)
    Start_Token = tensor(Start_Token_List)
    End_Token_List = ([[50256]] * Batch_size)
    End_Token = tensor(End_Token_List)
    Tag_code_tensor_dictionary['input_ids'] = torch.cat((Start_Token, Tag_code_tensor_dictionary['input_ids'], End_Token), dim=1)
    Attention_Token_List = ([[1]] * Batch_size)
    Attention_Token = tensor(Attention_Token_List)
    Tag_code_tensor_dictionary['attention_mask'] = torch.cat((Attention_Token, Tag_code_tensor_dictionary['attention_mask'], Attention_Token), dim=1)
    New_Problem_Text_Tensor = torch.cat((Total_Question_Text_Tensor_Dictionary['input_ids'], slow_code_tensor_dictionary['input_ids'], Tag_code_tensor_dictionary['input_ids']), dim=1)
    New_Problem_Attention_Tensor = torch.cat((Total_Question_Text_Tensor_Dictionary['attention_mask'], slow_code_tensor_dictionary['attention_mask'], Tag_code_tensor_dictionary['attention_mask']), dim=1)
    New_Problem_Text_Tensor_Dictionary = {'input_ids': New_Problem_Text_Tensor, 'attention_mask': New_Problem_Attention_Tensor}
    New_Tag_Tensor_List = [[]]
    New_Tag_Tensor_List[0].extend(([(- 100)] * len(Total_Question_Text_Tensor_Dictionary['input_ids'][0])))
    New_Tag_Tensor_List[0].extend(([(- 100)] * len(slow_code_tensor_dictionary['input_ids'][0])))
    New_Tag_Tensor = torch.LongTensor(New_Tag_Tensor_List)
    New_Tag_Tensor = torch.cat((New_Tag_Tensor, Tag_code_tensor_dictionary['input_ids']), dim=1)
    New_Tag_Tensor_Dictionary = {'input_ids': New_Tag_Tensor, 'attention_mask': New_Tag_Tensor}
    return [New_Problem_Text_Tensor_Dictionary, slow_code_tensor_dictionary, New_Tag_Tensor_Dictionary]

def The_train_function(Model, Devices, Training_Data, Test_Data, Multiple_GPUs, Command_line_parameters, Complementary_fill_function):
    Training_Data_Loader = DataLoader(dataset=Training_Data, batch_size=Command_line_parameters.batch_size, shuffle=True, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    Model.train()
    Total_number_of_steps = int((((Training_Data_Loader.__len__() * Command_line_parameters.epochs) / Command_line_parameters.batch_size) / Command_line_parameters.gradient_accumulation))
    Logger.info(f'================================ Total number of training steps: = {Total_number_of_steps} =========================')
    Optimizer = transformers.AdamW(Model.parameters(), lr=Command_line_parameters.lr, correct_bias=True)
    Logger.info('################################################## starting training ###############################')
    Cumulative_loss_per_gradient = 0
    for round in range(0, Command_line_parameters.epochs):
        Logger.info(f'========================================= training, epochs: {(round + 1)} =====================')
        Number_of_GPU_memory_runs_out = 0
        Training_loss_list = []
        for (Batch_Indexing, List_of_features) in enumerate(Training_Data_Loader):
            for (Index, tensor_dictionary) in enumerate(List_of_features):
                List_of_features[Index]['input_ids'] = tensor_dictionary['input_ids'].to(Devices)
                List_of_features[Index]['attention_mask'] = tensor_dictionary['attention_mask'].to(Devices)
                try:
                    List_of_features[Index]['token_type_ids'] = tensor_dictionary['token_type_ids'].to(Devices)
                except:
                    pass
            try:
                Model_Output = Model(List_of_features)
                Losses = Model_Output.loss
                Training_loss_list.append(Losses.item())
                if ((Batch_Indexing % Command_line_parameters.log_step) == 0):
                    Logger.info(f'& train, epochs: {(round + 1)} , (Total:{Command_line_parameters.epochs}), {(Batch_Indexing + 1)} (Total:{len(Training_Data_Loader)}), loss:{Losses:.3f}')
                if Multiple_GPUs:
                    Losses = Losses.mean()
                if (Command_line_parameters.gradient_accumulation > 1):
                    Losses = (Losses / Command_line_parameters.gradient_accumulation)
                Losses.backward()
                torch.nn.utils.clip_grad_norm_(Model.parameters(), Command_line_parameters.max_grad_norm)
                if (((Batch_Indexing + 1) % Command_line_parameters.gradient_accumulation) == 0):
                    Cumulative_loss_per_gradient += Losses.item()
                    Optimizer.step()
                    Optimizer.zero_grad()
            except RuntimeError as exception:
                if ('out of memory' in str(exception)):
                    Number_of_GPU_memory_runs_out += 1
                    Logger.info(f'#======== WARNING: ran out of memory,times: {Number_of_GPU_memory_runs_out} ========#')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    Logger.info(str(exception))
                    raise exception
        if (Command_line_parameters.task == 0):
            Model.Title_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Title_Layer/')
            Model.Problem_description_body_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Problem_description_body_Layer/')
            Model.Input_description_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Input_description_Layer/')
            Model.Output_description_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Output_description_Layer/')
            Model.IO_sample_testing_and_note_description_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/IO_sample_testing_and_note_description_Layer/')
            Model.Expert_Group_Integration_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Expert_Group_Integration_Layer/')
            torch.save(Model.MLP_enlarge_layer, f'{Command_line_parameters.Generated_models}/MLP_enlarge_layer.pkl')
            Model.IC_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/IC_layer/')
            Model.EC_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/EC_layer/')
            torch.save(Model.Multi_headed_attention_mechanism, f'{Command_line_parameters.Generated_models}/Multi_headed_attention_mechanism.pkl')
            Model.Final_output_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        elif (Command_line_parameters.task == 1):
            Model.Opened_expert_group_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Opened_expert_group_layer/')
            Model.Expert_Group_Integration_Layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Expert_Group_Integration_Layer/')
            torch.save(Model.MLP_enlarge_layer, f'{Command_line_parameters.Generated_models}/MLP_enlarge_layer.pkl')
            Model.IC_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/IC_layer/')
            Model.EC_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/EC_layer/')
            torch.save(Model.Multi_headed_attention_mechanism, f'{Command_line_parameters.Generated_models}/Multi_headed_attention_mechanism.pkl')
            Model.Final_output_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        elif (Command_line_parameters.task == 2):
            Model.Final_output_layer.save_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        Logger.info(f' train: epochs: {(round + 1)} , loss: {mean(Training_loss_list)}')
        Test_average_loss = Prediction_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, Complementary_fill_function)
    Logger.info(f'################## train: epochs: {(round + 1)}. End of round. Average loss from training: {mean(Training_loss_list)}. Test average loss: {Test_average_loss} ###################')
    Logger.info('##################################################################### End training ##################')
    prediction_generation_code_main((round + 1), Command_line_parameters.task)

def Prediction_functions(Model, Devices, Test_set_data, Multiple_GPUs, Command_line_parameters, Complementary_fill_function):
    Logger.info('=========================== starting Calculating predicted losses =========================')
    Model.eval()
    Test_data_loader = DataLoader(Test_set_data, batch_size=Command_line_parameters.batch_size, shuffle=True, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    with torch.no_grad():
        Loss_list = []
        for (Batch_Indexing, List_of_features) in enumerate(Test_data_loader):
            for (Index, tensor_dictionary) in enumerate(List_of_features):
                List_of_features[Index]['input_ids'] = tensor_dictionary['input_ids'].to(Devices)
                List_of_features[Index]['attention_mask'] = tensor_dictionary['attention_mask'].to(Devices)
                try:
                    List_of_features[Index]['token_type_ids'] = tensor_dictionary['token_type_ids'].to(Devices)
                except:
                    pass
            try:
                Model_Output = Model(List_of_features)
            except:
                continue
            Losses = Model_Output.loss
            Loss_list.append(Losses.item())
        Logger.info(f' evaluate test set loss:  {mean(Loss_list):.3f}')
        return mean(Loss_list)

def main():
    Command_line_parameters = Receive_command_line_arguments()
    global Logger
    Logger = Create_log_file_functions(Command_line_parameters)
    os.environ['CUDA_VISIBLE_DEVICES'] = Command_line_parameters.device
    Devices = ('cuda' if Command_line_parameters.cuda else 'cpu')
    Logger.info('using :{}'.format(Devices))
    if Command_line_parameters.seed:
        Setting_random_seed_functions(Command_line_parameters)
    if (not os.path.exists(Command_line_parameters.log_path)):
        os.mkdir(Command_line_parameters.log_path)
    if (not os.path.exists(Command_line_parameters.Generated_models)):
        os.mkdir(Command_line_parameters.Generated_models)
    if (Command_line_parameters.task == 0):
        Model = E_code_Model(Command_line_parameters)
    elif (Command_line_parameters.task == 1):
        Model = No_expert_Model(Command_line_parameters)
    elif (Command_line_parameters.task == 2):
        Model = GPT_Neo_Model(Command_line_parameters)
    Model.to(Devices)
    Multiple_GPUs = False
    Total_number_of_parameters = 0
    List_of_model_parameters = Model.parameters()
    for (layer, Parameters_of_a_layer) in enumerate(List_of_model_parameters):
        Total_number_of_parameters += Parameters_of_a_layer.numel()
    Logger.info(f'==============================Total model parameters: {Total_number_of_parameters} ===============================')
    Logger.info('======================Load training data.... ==========================')
    if (Command_line_parameters.task == 0):
        Training_Data = Train_Dataset()
        Test_Data = Test_Dataset()
        The_train_function(Model, Devices, Training_Data, Test_Data, Multiple_GPUs, Command_line_parameters, E_code_complementary_fill_function)
    elif (Command_line_parameters.task == 1):
        Training_Data = No_expert_Train_Dataset()
        Test_Data = No_expert_Test_Dataset()
        The_train_function(Model, Devices, Training_Data, Test_Data, Multiple_GPUs, Command_line_parameters, No_expert_completion_fill_function)
    elif (Command_line_parameters.task == 2):
        Training_Data = GPT_Neo_Train_Dataset()
        Test_Data = GPT_Neo_Test_Dataset()
        The_train_function(Model, Devices, Training_Data, Test_Data, Multiple_GPUs, Command_line_parameters, GPT_complementary_padding_function)
if (__name__ == '__main__'):
    main()