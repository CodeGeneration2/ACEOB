# coding=utf-8

import os
from numpy import mean

import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
# from torch.utils.tensorboard import SummaryWriter
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


# ##################################################################### args ######################################
def setup_train_args():

    # ----------------------------------- 命令行解析模块 -----------#
    parser = argparse.ArgumentParser()


    parser.add_argument('--topk', default=1, type=int, required=False, help='最高k选1，默认8')
    parser.add_argument('--topp', default=0.5, type=float, required=False, help='最高积累概率，默认0.9')  # 0

    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')  # 0,1
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')

    parser.add_argument('--arch', default="EleutherAI/gpt-neo-125M", help='model参数')


    parser.add_argument('--log_path', default='log/Prediction_Log.txt', type=str, required=False, help='预测logger存放位置')

    parser.add_argument('--epochs', default=1, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')  # 8
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    parser.add_argument('--Generated_models', default='Generated_models', type=str, required=False, help='Generated_models')
    parser.add_argument('--Whether_to_use_trained_local_mods', default=1, type=str, required=False, help='0为否，1为是')

    parser.add_argument('--seed', type=int, default=666, help='设置种子用于生成随机数，以使得训练的结果是确定的')  # None
    parser.add_argument('--num_workers', type=int, default=5, help="dataloaderload数据时使用的线程数量")

    # --------------------------- 解析参数 ------------------#
    args = parser.parse_args()

    return args


# ##################################################################### 随机种子函数 ####################################
def set_random_seed_function(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ##################################################################### logger文件 ######################################
def create_log_file_function(args):
    """
    将loggeroutput到logger文件和console
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ----------------------- 格式 ---------------------#
    time_Format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ------------------- 创建一个handler，用于写入 logger文件 ---------------#
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(time_Format)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # ------------------- 创建一个handler，用于将loggeroutput到 console（console） -----------#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(time_Format)
    logger.addHandler(console)

    return logger


# ##################################################################### 创建model ######################################
def create_model(args):

    model = Model(args)

    # --------------------------------- 根据tokenizer的vocabulary 调整 GPT2model的voca的大小 --------#
    # model.resize_token_embeddings(词表词数)

    return model



def complementary_fill_function(A_batch_data):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param A_batch_data:
    :return:
    title,problem_Description_Subject,Input_Description,Output_Description,Input_output_sample_tests_and_Note_description, certain_slow_code, label_code
    """
    # ---------------------------------------------------------- 转化为张量字典 --------------------------------#
    code_dictionary = Tokenized(A_batch_data[0], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return code_dictionary


# ##################################################################### 训练函数 #######################################
def Prediction_generation_function(model, devices, Dataset, Multiple_GPUs, args, Training_set_or_test_set):

    # ---------------------------- 将 train_data 为 train_dataloader --------------------#
    dataloader = DataLoader(dataset=Dataset,  # 使用的Dataset
                         batch_size=args.batch_size,  # 批处理样本大小
                         shuffle=False,  # 每次迭代前不打乱数据
                         num_workers=args.num_workers,  # 使用一个进程
                         collate_fn=complementary_fill_function
                         )

    # -------- 进入预测状态：不计算梯度 --------#
    model.eval()

    # --------------------------------------- 写入logger --------------------------------#
    logger.info('############################# 开始生成：starting training ############################')


    # ############################################## 开始 ########################################## #
    # =========================== 无梯度 ======================#
    with torch.no_grad():
        # ---------------------------------- 计算Bleu ----------------------------------------#
        Total_list_running_times = []

        # ------------------- dataloader循环迭代 -------------#
        # ---------- Feature_List：title,problem_Description_Subject,Input_Description,Output_Description,Input_output_sample_tests_and_Note_description, certain_slow_code, label_code -------#
        for Batch_index, code_dictionary in enumerate(dataloader):

            # ----------------------------- 移动到GPU -------------------------------#
            code_dictionary["input_ids"] = code_dictionary["input_ids"].to(devices)
            code_dictionary["attention_mask"] = code_dictionary["attention_mask"].to(devices)
            try:
                code_dictionary["token_type_ids"] = code_dictionary["token_type_ids"].to(devices)
            except:
                pass

            # -------------------------------- 传入GPT-2model ----------------#
            model_output = model(code_dictionary)

            Predicted_running_time = model_output.cpu().numpy().tolist()
            Predicted_running_time = Predicted_running_time[0][0]

            print(type(Predicted_running_time))

            Total_list_running_times.append(Predicted_running_time)
            logger.info(f'\033[0:34m======================= 批次：{Batch_index}. model_output：{Predicted_running_time}   ========\033[m')

            break


    Average_running_time = mean(Total_list_running_times)

    with open(f"Forecast time.txt", 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_running_times))

    # ------------------------------------------ 结束：写入logger ------------------------------------------------#
    logger.info(f'\033[0:34m======================= Average_running_time：{Average_running_time}    结束 ========\033[m')

    return Average_running_time


# ####################################################################### 主函数 #######################################
def main():
    # -------------------------- 初始化args -----#
    args = setup_train_args()

    # -------------------------------- 创建logger对象 -----#
    # ----------- logger同时output到文件和console -------#
    global logger
    logger = create_log_file_function(args)

    # ================================================================ 当用户使用GPU,并且GPU可用时 ===================#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    devices = 'cuda' if args.cuda else 'cpu'
    # devices = 'cpu'
    logger.info('using devices:{}'.format(devices))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed_function(args)

    # -------------------------------------------------------- 初始化tokenizer --------------#

    # -------------------- 创建model的output目录 -------------------#
    if not os.path.exists(args.Generated_models):
        os.mkdir(args.Generated_models)

    # --------------------- loadGPTmodel ------------------------#
    model = create_model(args)

    # ------------- 移到GPU ----#
    model.to(devices)

    # -------------------------- 是否使用Multiple_GPUs进行并行运算 ------------------------#
    Multiple_GPUs = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use 多个 GPUs to 训练")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])

    # ------------------------- 记录model参数数量 ---------------#
    Total_number_parameters = 0
    model_parameter_list = model.parameters()
    for a_layer_parameters in model_parameter_list:
        Total_number_parameters += a_layer_parameters.numel()
    # ------------------------------------ log ----------------#
    logger.info(f'=========== number of model model_parameter_list: {Total_number_parameters} =================')

    # -------------------------------- log ---------#
    logger.info("=================== loadtrain_data。。。 =======================")

    # ============================== load数据 ===============================#
    forecast_Data = Predicted_Dataset_Class()

    # ----------------------------- 开始训练 ----------------------------#
    Training_set_BLEU_score = Prediction_generation_function(model, devices, forecast_Data, Multiple_GPUs, args,"Training_set")

    # ---------------------------------------------------- Final_Output ---------------------------------------------------#
    logger.info(f'\033[0:34m======= Training_set_BLEU_score：{Training_set_BLEU_score}  ===============\033[m')

# ############################################################ entrance ####################################################
if __name__ == '__main__':
    main()

class Predictive_code_generation_class():
    def entrance(self):
        main()
