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

from model import Modle
from train_Dataset import TrainDatasetClass
from test_Dataset import Test_set_Dataset_class
from sacrebleu.metrics import BLEU, CHRF, TER


Tokenized = GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")

# ##################################################################### args ######################################
def setup_train_args():

    # ----------------------------------- 命令行解析模块 -----------#
    parser = argparse.ArgumentParser()

    parser.add_argument('--topk', default=4, type=int, required=False, help='最高k选1，默认8')
    parser.add_argument('--topp', default=0.8, type=float, required=False, help='最高积累概率，默认0.9')  # 0
    parser.add_argument('--temperature', default=0.25, type=float, required=False, help='最高积累概率，默认0.9')  # 0

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
    将logger输出到logger文件和console
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

    # ------------------- 创建一个handler，用于将logger输出到 console（console） -----------#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(time_Format)
    logger.addHandler(console)

    return logger


# ##################################################################### 创建model ######################################
def create_model(args):

    model = Modle(args)

    # --------------------------------- 根据tokenizer的vocabulary 调整 GPT2model的voca的大小 --------#
    # model.resize_token_embeddings(词表词数)

    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 使用 top-k 和/或 nucleus (top-p) 过滤 对 logit 分布进行过滤
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # ----------------- 现在将批量设为1 --- 可以更新更多，但代码将不太清楚 -------------#
    assert logits.dim() == 1

    # ------------------------------- 安全检查 ---------- Safety check ------------#
    top_k = min(top_k, logits.size(-1))

    # -------------------------------------- TopK 方法 -------------------------------#
    if top_k > 0:
        # --------------- torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices) ---#
        TopK_Return_Value = torch.topk(logits, top_k)

        # --------------- ...表示其他维度由计算机自行推断 -----------------------------------------#
        # ---------------- 增加了一个维度。newaxis效果和None是一样的，None是别名 --------------------#
        TopK_last_token_probability = TopK_Return_Value[0][..., -1, None]

        # -------------------------------- 移除所有概率小于top-k最后一个token的token --------------#
        index_removes_Boolean_matrix = logits < TopK_last_token_probability

        # -------------------------- 对于topk之外的其他元素的logits值设为filter_value ---------------------#
        logits[index_removes_Boolean_matrix] = filter_value

    # -------------------------------------- TopP 方法 -------------------------------#
    if top_p > 0.0:
        # ------------------------------- 对logits进行递减排序 ---------------------------#
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # -------------------------------- Cumulative_probability ----------------#
        Cumulative_probability = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # ---------------------------- 删除 累积概率 高于 阈值 的 Token --------------------------#
        filtering_index_removes_Boolean_matrix = Cumulative_probability > top_p

        # ----------------------- 将index向右移动，使第一个标记也保持在阈值以上 ----------------------#
        filtering_index_removes_Boolean_matrix[..., 1:] = filtering_index_removes_Boolean_matrix[..., :-1].clone()
        filtering_index_removes_Boolean_matrix[..., 0] = 0

        index_remove_id_matrix = sorted_indices[filtering_index_removes_Boolean_matrix]

        # -------------------------- 对于 核方法 之外 的其他元素的logits值设为 filter_value ---------------------#
        logits[index_remove_id_matrix] = filter_value

    return logits


def complementary_fill_function(A_batch_data):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param A_batch_data:
    :return:
    title,problem_Description_Subject,Input_Description,Output_Description,Input_output_sample_tests_and_Note_description, certain_slow_code, label_code
    """
    # ---------------------------------------------------------- 转化为张量字典 --------------------------------#
    Question_text_dictionary = Tokenized(A_batch_data[0][0], max_length=512, truncation=True, padding=True, return_tensors="pt")
    certain_slow_code_dictionary = Tokenized(A_batch_data[0][1], max_length=768, truncation=True, padding=True, return_tensors="pt")

    New_problem_text_tensor = torch.cat((Question_text_dictionary["input_ids"], certain_slow_code_dictionary["input_ids"]), dim=1)
    New_Problem_Attention_Tensor = torch.cat((Question_text_dictionary["attention_mask"], certain_slow_code_dictionary["attention_mask"]), dim=1)

    New_problem_text_tensor_dictionary = {"input_ids":New_problem_text_tensor,"attention_mask":New_Problem_Attention_Tensor}


    return [New_problem_text_tensor_dictionary, certain_slow_code_dictionary], A_batch_data[0][2], A_batch_data[0][3]


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
        total_BLEU_scores_list = []

        # ------------------- dataloader循环迭代 -------------#
        # ---------- Feature_List：title,problem_Description_Subject,Input_Description,Output_Description,Input_output_sample_tests_and_Note_description, certain_slow_code, label_code -------#
        for Batch_index, (Feature_List, label_code, Data_original_path) in enumerate(dataloader):

            # ----------------------------- 移动到GPU -------------------------------#
            for index, A_certain_dictionary in enumerate(Feature_List):
                Feature_List[index]["input_ids"] = A_certain_dictionary["input_ids"].to(devices)
                try:
                    Feature_List[index]["attention_mask"] = A_certain_dictionary["attention_mask"].to(devices)
                    Feature_List[index]["token_type_ids"] = A_certain_dictionary["token_type_ids"].to(devices)
                except:
                    pass

            # ----------------------------- 移动到GPU -------------------------------#
            Generated_code = tensor([[102]])
            Generated_code_attention_matrix = tensor([[1]])

            Feature_List.append({"input_ids":Generated_code.to(devices),"attention_mask":Generated_code_attention_matrix.to(devices)})

            # ---------------------------------------------
            Generated_list = []

            # ####################################### 最多生成max_len个token #################################
            for _ in range(len(Feature_List[1]["input_ids"][0])):

                # -------------------------------- 传入GPT-2model ----------------#
                model_output = model(Feature_List).logits

                # ----------------------- model_output[0]: model的预测结果序列 -------------------------#
                # ----------------------- model_output[0][-1, :]: Mod_prediction_probability_next_Token -----------#
                Mod_prediction_probability_next_Token = model_output[0,-1, :]

                # ------------- 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率 -------#
                for id in set(Generated_list):  # set 集合的意思
                    Mod_prediction_probability_next_Token[id] /= 1

                # ------------------------------------ 温度：处理预测概率 -----------------------------#
                Mod_prediction_probability_next_Token = Mod_prediction_probability_next_Token / args.temperature
                #Mod_prediction_probability_next_Token = Mod_prediction_probability_next_Token / 1

                # ----------- 对于[UNK]的概率设为无穷小，也就是说model的预测结果不可能是[UNK]这个token -----------#
                Mod_prediction_probability_next_Token[102] = -float('Inf')
                #Mod_prediction_probability_next_Token[0] = -float('Inf')

                # -------------------------------------- 过滤 --------------------------------#
                filtering_logits = top_k_top_p_filtering(Mod_prediction_probability_next_Token, top_k=args.topk, top_p=args.topp)

                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                Predictive_Token = torch.multinomial(F.softmax(filtering_logits, dim=-1), num_samples=1)

                # ------------------------------ 遇到[SEP]则表明response生成结束 --------------#
                if Predictive_Token == 50256:
                    break

                # ------------------------------- 加入Generated_list ----------------#
                Generated_list.append(Predictive_Token.item())

                # -------------------------------- model生成的部分Token 和 原本的输入文本进行拼接 ------#
                Feature_List[-1]["input_ids"] = torch.cat((Feature_List[-1]["input_ids"], tensor([[Predictive_Token]]).to(devices)), dim=1)
                Feature_List[-1]["attention_mask"] = torch.cat((Feature_List[-1]["attention_mask"], tensor([[1]]).to(devices)), dim=1)

            # ------------------------------ 转化为 Output_Text -------------------------#
            Output_Text = Tokenized.batch_decode(Feature_List[-1]["input_ids"])[0].replace("[CLS]","")

            # -------------------------------- 计算Bleu分数 --------------------------------------------#
            Standard_answer_list = [
                [label_code],
            ]
            model_generation_list = [Output_Text]
            bleu = BLEU()
            bleu_score = bleu.corpus_score(model_generation_list,Standard_answer_list).score
            total_BLEU_scores_list.append(bleu_score)

            # ----------------------------- 保存记录文件 -------------------------------------#
            with open(f"Generated_code/{Training_set_or_test_set}/{Batch_index},BLEU_score,{bleu_score:.3f}.txt", 'w', encoding='UTF-8') as f:
                f.write(Output_Text)
            # ------------------------------ 保存记录文件 ------------------------------------#
            with open(f"Generated_code/{Training_set_or_test_set}/{Batch_index},Standard_answer.txt", 'w', encoding='UTF-8') as f:
                f.write(label_code)

            # ------------------------------------------ 更新logger信息 -------------------------------------------------#
            logger.info(f'\033[0:32m{Training_set_or_test_set}，第 {Batch_index} 条数据(共{len(dataloader)}批), 其BLEU_score为：{bleu_score:.3f}, 其慢速代码长度：{len(Feature_List[-2]["input_ids"][0])}, 已经预测生成\033[m')


    Average_BLEU_score = mean(total_BLEU_scores_list)
    # ------------------------------ 保存记录文件 ------------------------------------#
    with open(f"Generated_code/{Training_set_or_test_set}/0,BLEU_score列表,Average_BLEU_score,{Average_BLEU_score:.3f}.txt", 'w', encoding='UTF-8') as f:
        f.write(str(total_BLEU_scores_list))

    # ------------------------------------------ 结束：写入logger ------------------------------------------------#
    logger.info(f'\033[0:34m======================= {Training_set_or_test_set}总BLEU_score为：{Average_BLEU_score:.3f}。 {Training_set_or_test_set}结束 ========\033[m')

    return Average_BLEU_score


# ####################################################################### 主函数 #######################################
def main():
    # -------------------------- 初始化args -----#
    args = setup_train_args()

    # -------------------------------- 创建logger对象 -----#
    # ----------- logger同时输出到文件和console -------#
    global logger
    logger = create_log_file_function(args)

    # ================================================================ 当用户使用GPU,并且GPU可用时 ===================#
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

    # -------------------- 创建model的输出目录 -------------------#
    if not os.path.exists(args.Generated_models):
        os.mkdir(args.Generated_models)

    # -------------------- 创建model Generated_code 的输出目录 -------------------#
    if not os.path.exists("Generated_code/test_code"):
        os.mkdir("Generated_code/test_code")


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
    test_data = Test_set_Dataset_class()

    # ----------------------------- 开始训练 ----------------------------#
    Test_set_BLEU_scores = Prediction_generation_function(model, devices, test_data, Multiple_GPUs, args,"test_code")

    # ---------------------------------------------------- Final_Output ---------------------------------------------------#
    logger.info(f'\033[0:34m======= Test_set_BLEU_scores：{Test_set_BLEU_scores} ===============\033[m')

# ############################################################ entrance ####################################################
if __name__ == '__main__':
    main()

class Predictive_code_generation_class():
    def entrance(self):
        main()
