import os
from numpy import mean
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from Model import E_code_Model
from No_expert_Model import No_expert_Model
from GPT_Neo_Model import GPT_Neo_Model
from train_Dataset import Train_Dataset
from GPT_Neo_train_Dataset import GPT_Neo_Train_Dataset
from test_Dataset import Test_Dataset
from GPT_Neo_test_Dataset import GPT_Neo_Test_Dataset
from No_expert_train_Dataset import No_expert_Train_Dataset
from No_expert_test_Dataset import No_expert_Test_Dataset
from CodeBleu import _bleu
from sacrebleu.metrics import BLEU, CHRF, TER
from IO_testing_of_generated_code import main as IO_testing_of_generated_code_main
projection_log = None
Tokenized_GPT_word_list = GPT2Tokenizer.from_pretrained('GPT_Tokenizer/', pad_token='[PAD]', cls_token='[CLS]')
Tokenized_Bert_word_list = BertTokenizer.from_pretrained('Bert_Tokenizer/', pad_token='[PAD]')

def Receive_command_line_arguments():
    Command_line_parser = argparse.ArgumentParser()
    Command_line_parser.add_argument('--device', default='0', type=str, required=False)
    Command_line_parser.add_argument('--task', default=0, type=int, required=False, help='0: E-code, 1: No-expert group, 2: GPT-Neo')
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

def Setting_random_seed_functions(Command_line_parameters):
    torch.manual_seed(Command_line_parameters.seed)
    random.seed(Command_line_parameters.seed)
    np.random.seed(Command_line_parameters.seed)
    if Command_line_parameters.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def Create_log_file_functions(Command_line_parameters):
    Prediction_log = logging.getLogger(__name__)
    Prediction_log.setLevel(logging.INFO)
    Time_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    Log_file_writers = logging.FileHandler(filename=Command_line_parameters.log_path)
    Log_file_writers.setFormatter(Time_format)
    Log_file_writers.setLevel(logging.INFO)
    Prediction_log.addHandler(Log_file_writers)
    Console = logging.StreamHandler()
    Console.setLevel(logging.DEBUG)
    Console.setFormatter(Time_format)
    Prediction_log.addHandler(Console)
    return Prediction_log

def TopK_and_kernel_method_filter_functions(Prediction_probability_of_the_next_Token, TopK=0, TopP=0.0, Negative_Infinity=(- float('Inf'))):
    assert (Prediction_probability_of_the_next_Token.dim() == 1)
    TopK = min(TopK, Prediction_probability_of_the_next_Token.size((- 1)))
    if (TopK > 0):
        TopK_return_value = torch.topk(Prediction_probability_of_the_next_Token, TopK)
        Probability_of_the_last_token_of_TopK = TopK_return_value[0][(..., (- 1), None)]
        Index_removal_Boolean_matrix = (Prediction_probability_of_the_next_Token < Probability_of_the_last_token_of_TopK)
        Prediction_probability_of_the_next_Token[Index_removal_Boolean_matrix] = Negative_Infinity
    if (TopP > 0.0):
        (Sort_Probability, Sorted_Indexes) = torch.sort(Prediction_probability_of_the_next_Token, descending=True)
        Cumulative_probability = torch.cumsum(F.softmax(Sort_Probability, dim=(- 1)), dim=(- 1))
        Index_removal_boolean_matrix_after_sorting = (Cumulative_probability > TopP)
        Index_removal_boolean_matrix_after_sorting[..., 1:] = Index_removal_boolean_matrix_after_sorting[..., :(- 1)].clone()
        Index_removal_boolean_matrix_after_sorting[(..., 0)] = 0
        Index_removal_id_matrix = Sorted_Indexes[Index_removal_boolean_matrix_after_sorting]
        Prediction_probability_of_the_next_Token[Index_removal_id_matrix] = Negative_Infinity
    return Prediction_probability_of_the_next_Token

def E_code_complementary_fill_function(batch_of_data):
    Title_tensor_dictionary = Tokenized_Bert_word_list(batch_of_data[0][0], max_length=512, truncation=True, padding=True, return_tensors='pt')
    Question_description_subject_tensor_dictionary = Tokenized_Bert_word_list(batch_of_data[0][1], max_length=512, truncation=True, padding=True, return_tensors='pt')
    Input_Description_Tensor_Dictionary = Tokenized_Bert_word_list(batch_of_data[0][2], max_length=512, truncation=True, padding=True, return_tensors='pt')
    Output_description_tensor_dictionary = Tokenized_Bert_word_list(batch_of_data[0][3], max_length=512, truncation=True, padding=True, return_tensors='pt')
    Input_and_output_sample_tests_and_Note_description_tensor_dictionary = Tokenized_Bert_word_list(batch_of_data[0][4], max_length=512, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(batch_of_data[0][5], max_length=768, truncation=True, padding=True, return_tensors='pt')
    return ([Title_tensor_dictionary, Question_description_subject_tensor_dictionary, Input_Description_Tensor_Dictionary, Output_description_tensor_dictionary, Input_and_output_sample_tests_and_Note_description_tensor_dictionary, slow_code_tensor_dictionary], batch_of_data[0][6], batch_of_data[0][7], batch_of_data[0][8])

def No_expert_completion_fill_function(batch_of_data):
    Total_Question_Text_Tensor_Dictionary = Tokenized_Bert_word_list(batch_of_data[0][0], max_length=2048, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(batch_of_data[0][1], max_length=768, truncation=True, padding=True, return_tensors='pt')
    return ([Total_Question_Text_Tensor_Dictionary, slow_code_tensor_dictionary], batch_of_data[0][2], batch_of_data[0][3], batch_of_data[0][4])

def GPT_complementary_padding_function(batch_of_data):
    Question_Text_Tensor_Dictionary = Tokenized_Bert_word_list(batch_of_data[0][0], max_length=512, truncation=True, padding=True, return_tensors='pt')
    slow_code_tensor_dictionary = Tokenized_GPT_word_list(batch_of_data[0][1], max_length=768, truncation=True, padding=True, return_tensors='pt')
    New_Problem_Text_Tensor = torch.cat((Question_Text_Tensor_Dictionary['input_ids'], slow_code_tensor_dictionary['input_ids']), dim=1)
    New_Problem_Attention_Tensor = torch.cat((Question_Text_Tensor_Dictionary['attention_mask'], slow_code_tensor_dictionary['attention_mask']), dim=1)
    New_Problem_Text_Tensor_Dictionary = {'input_ids': New_Problem_Text_Tensor, 'attention_mask': New_Problem_Attention_Tensor}
    return ([New_Problem_Text_Tensor_Dictionary, slow_code_tensor_dictionary], batch_of_data[0][2], batch_of_data[0][3], batch_of_data[0][4])

def Prediction_generation_functions(Model, Devices, Dataset, Multiple_GPUs, Command_line_parameters, Training_set_or_test_set, Number_of_rounds1, Complementary_fill_function, Task):
    Data_Loader = DataLoader(dataset=Dataset, batch_size=Command_line_parameters.batch_size, shuffle=False, num_workers=Command_line_parameters.num_workers, collate_fn=Complementary_fill_function)
    Model.eval()
    projection_log.info('############################################## starting Prediction_generation_code ###########')
    with torch.no_grad():
        Total_list_of_BLEU_scores = []
        Total_list_of_CodeBLEU_scores = []
        Total_list_of_compilation_rates = []
        Total_list_of_IO_test_pass_rates = []
        for (Batch_Indexing, (List_of_features, The_tag_code, Input_and_output_case_paths, Data_original_path)) in enumerate(Data_Loader):
            if ((Training_set_or_test_set == 'Train') and ((Batch_Indexing % Command_line_parameters.Train_set_interval) != 0)):
                continue
            elif ((Training_set_or_test_set == 'Test') and ((Batch_Indexing % Command_line_parameters.Test_set_interval) != 0)):
                continue
            else:
                for (Index, tensor_dictionary) in enumerate(List_of_features):
                    List_of_features[Index]['input_ids'] = tensor_dictionary['input_ids'].to(Devices)
                    List_of_features[Index]['attention_mask'] = tensor_dictionary['attention_mask'].to(Devices)
                    try:
                        List_of_features[Index]['token_type_ids'] = tensor_dictionary['token_type_ids'].to(Devices)
                    except:
                        pass
                Generated_Code = tensor([[102]])
                Generated_code_attention_matrix = tensor([[1]])
                List_of_features.append({'input_ids': Generated_Code.to(Devices), 'attention_mask': Generated_code_attention_matrix.to(Devices)})
                Generated_list = []
                if (Command_line_parameters.Maximum_length_pattern_of_generated_code == 0):
                    Maximum_length = Command_line_parameters.Maximum_length_of_generated_code
                else:
                    Maximum_length = len(List_of_features[5]['input_ids'][0])
                for _ in range(Maximum_length):
                    Model_Output = Model(List_of_features).logits
                    Model_predicted_probability_of_the_next_Token = Model_Output[0, (- 1), :]
                    for id in set(Generated_list):
                        Model_predicted_probability_of_the_next_Token[id] /= 1
                    Model_predicted_probability_of_the_next_Token = (Model_predicted_probability_of_the_next_Token / Command_line_parameters.Temperature)
                    Model_predicted_probability_of_the_next_Token[102] = (- float('Inf'))
                    Predicted_probability_of_the_next_Token_after_filtering = TopK_and_kernel_method_filter_functions(Model_predicted_probability_of_the_next_Token, TopK=Command_line_parameters.topk, TopP=Command_line_parameters.topp)
                    Predicted_Token = torch.multinomial(F.softmax(Predicted_probability_of_the_next_Token_after_filtering, dim=(- 1)), num_samples=1)
                    if (Predicted_Token == 50256):
                        break
                    Generated_list.append(Predicted_Token.item())
                    if (Task == 0):
                        List_of_features[(- 1)]['input_ids'] = torch.cat((List_of_features[(- 1)]['input_ids'], tensor([[Predicted_Token]]).to(Devices)), dim=1)
                        List_of_features[(- 1)]['attention_mask'] = torch.cat((List_of_features[(- 1)]['attention_mask'], tensor([[1]]).to(Devices)), dim=1)
                    elif (Task == 0):
                        List_of_features[(- 1)]['input_ids'] = torch.cat((List_of_features[(- 1)]['input_ids'], tensor([[Predicted_Token]]).to(Devices)), dim=1)
                        List_of_features[(- 1)]['attention_mask'] = torch.cat((List_of_features[(- 1)]['attention_mask'], tensor([[1]]).to(Devices)), dim=1)
                    elif (Task == 2):
                        List_of_features[0]['input_ids'] = torch.cat((List_of_features[0]['input_ids'], tensor([[Predicted_Token]]).to(Devices)), dim=1)
                        List_of_features[0]['attention_mask'] = torch.cat((List_of_features[0]['attention_mask'], tensor([[1]]).to(Devices)), dim=1)
                        List_of_features[(- 1)]['input_ids'] = torch.cat((List_of_features[(- 1)]['input_ids'], tensor([[Predicted_Token]]).to(Devices)), dim=1)
                        List_of_features[(- 1)]['attention_mask'] = torch.cat((List_of_features[(- 1)]['attention_mask'], tensor([[1]]).to(Devices)), dim=1)
                Output_Text = Tokenized_GPT_word_list.batch_decode(List_of_features[(- 1)]['input_ids'])[0].replace('[CLS]', '')
                Standard_answer_list = [[The_tag_code]]
                Model_generation_list = [Output_Text]
                bleu = BLEU()
                bleu_score = bleu.corpus_score(Model_generation_list, Standard_answer_list).score
                Total_list_of_BLEU_scores.append(bleu_score)
                try:
                    Codebleu_score = round(_bleu(The_tag_code, Output_Text), 2)
                except:
                    Codebleu_score = 0
                Total_list_of_CodeBLEU_scores.append(Codebleu_score)
                IO_case_path = Input_and_output_case_paths.replace('/', '%')
                with open(f'Generated_code/Round_{Number_of_rounds1}_prediction_code/{Training_set_or_test_set}/{Batch_Indexing},CodeBLEU:{Codebleu_score:.3f},IO_Paths:{IO_case_path}.txt', 'w', encoding='UTF-8') as f:
                    f.write(Output_Text)
                projection_log.info(f'Prediction_generation_code, {Training_set_or_test_set}, {Batch_Indexing}(Total:{len(Data_Loader)}), CodeBLEU:{Codebleu_score:.3f}, BLEU:{bleu_score:.3f}')
    Average_BLEU_score = mean(Total_list_of_BLEU_scores)
    Average_CodeBLEU_score = mean(Total_list_of_CodeBLEU_scores)
    with open(f'Generated_code/Round_{Number_of_rounds1}_prediction_code/BLEU_Score_List,Average:{Average_BLEU_score:.3f}.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_BLEU_scores))
    with open(f'Generated_code/Round_{Number_of_rounds1}_prediction_code/CodeBLEU_Score_List,Average:{Average_CodeBLEU_score:.3f}.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_CodeBLEU_scores))
    projection_log.info(f'================ Prediction_generation_code, {Training_set_or_test_set}, CodeBLEU:{Average_CodeBLEU_score:.3f}, BLEU:{Average_BLEU_score:.3f} ==')
    return (Average_CodeBLEU_score, Average_BLEU_score)

def main(Number_of_rounds2, task):
    Command_line_parameters = Receive_command_line_arguments()
    global projection_log
    projection_log = Create_log_file_functions(Command_line_parameters)
    os.environ['CUDA_VISIBLE_DEVICES'] = Command_line_parameters.device
    Devices = ('cuda' if Command_line_parameters.cuda else 'cpu')
    projection_log.info('using:{}'.format(Devices))
    if Command_line_parameters.seed:
        Setting_random_seed_functions(Command_line_parameters)
    if (not os.path.exists(f'Generated_code/Round_{Number_of_rounds2}_prediction_code/Train')):
        os.makedirs(f'Generated_code/Round_{Number_of_rounds2}_prediction_code/Train')
    if (not os.path.exists(f'Generated_code/Round_{Number_of_rounds2}_prediction_code/Test')):
        os.makedirs(f'Generated_code/Round_{Number_of_rounds2}_prediction_code/Test')
    if (task == (- 1)):
        if (Command_line_parameters.task == 0):
            Model = E_code_Model(Command_line_parameters)
        elif (Command_line_parameters.task == 1):
            Model = No_expert_Model(Command_line_parameters)
        elif (Command_line_parameters.task == 2):
            Model = GPT_Neo_Model(Command_line_parameters)
    elif (task == 0):
        Model = E_code_Model(Command_line_parameters)
    elif (task == 1):
        Model = No_expert_Model(Command_line_parameters)
    elif (task == 2):
        Model = GPT_Neo_Model(Command_line_parameters)
    Model.to(Devices)
    Multiple_GPUs = False
    Total_number_of_parameters = 0
    List_of_model_parameters = Model.parameters()
    for Parameters_of_a_layer in List_of_model_parameters:
        Total_number_of_parameters += Parameters_of_a_layer.numel()
    projection_log.info(f'================When tested Model Total number of model parameters : {Total_number_of_parameters} ====================')
    if (task == (- 1)):
        if (Command_line_parameters.task == 0):
            Training_Data = Train_Dataset()
            Test_Data = Test_Dataset()
            (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, E_code_complementary_fill_function, Command_line_parameters.task)
            (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, E_code_complementary_fill_function, Command_line_parameters.task)
        elif (Command_line_parameters.task == 1):
            Training_Data = No_expert_Train_Dataset()
            Test_Data = No_expert_Test_Dataset()
            (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, No_expert_completion_fill_function, Command_line_parameters.task)
            (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, No_expert_completion_fill_function, Command_line_parameters.task)
        elif (Command_line_parameters.task == 2):
            Training_Data = GPT_Neo_Train_Dataset()
            Test_Data = GPT_Neo_Test_Dataset()
            (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, GPT_complementary_padding_function, Command_line_parameters.task)
            (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, GPT_complementary_padding_function, Command_line_parameters.task)
    elif (task == 0):
        Training_Data = Train_Dataset()
        Test_Data = Test_Dataset()
        (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, E_code_complementary_fill_function, task)
        (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, E_code_complementary_fill_function, task)
    elif (task == 1):
        Training_Data = No_expert_Train_Dataset()
        Test_Data = No_expert_Test_Dataset()
        (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, No_expert_completion_fill_function, Command_line_parameters.task)
        (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, No_expert_completion_fill_function, Command_line_parameters.task)
    elif (task == 2):
        Training_Data = GPT_Neo_Train_Dataset()
        Test_Data = GPT_Neo_Test_Dataset()
        (Training_set_CodeBLEU_score, Training_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Training_Data, Multiple_GPUs, Command_line_parameters, 'Train', Number_of_rounds2, GPT_complementary_padding_function, task)
        (Test_set_CodeBLEU_score, Test_set_BLEU_score) = Prediction_generation_functions(Model, Devices, Test_Data, Multiple_GPUs, Command_line_parameters, 'Test', Number_of_rounds2, GPT_complementary_padding_function, task)
    projection_log.info(f'================= Prediction_generation_code: epochs {Number_of_rounds2} : Train_CodeBLEU: {Training_set_CodeBLEU_score}  ,  Train_BLEU: {Training_set_BLEU_score} ==')
    projection_log.info(f'================= Prediction_generation_code: epochs {Number_of_rounds2} : Test_CodeBLEU: {Test_set_CodeBLEU_score}   ,   Test_BLEU: {Test_set_BLEU_score} ==')
    projection_log.info('######################################## End Prediction_generation_code #####################')
    IO_testing_of_generated_code_main()
if (__name__ == '__main__'):
    main('0', (- 1))