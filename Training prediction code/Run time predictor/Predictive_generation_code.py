# coding=utf-8

import os
from numpy import mean
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import wandb
# wandb.login(key="82cfc67dbc70c15d489f7eaaae135311f3716c1a")

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

from model import 模型类
from predicted_time_Dataset import 训练集Dataset类

from sacrebleu.metrics import BLEU, CHRF, TER

# 网址：https://www.bilibili.com/video/BV1La4y1Y7ug?spm_id_from=333.337.search-card.all.click

日志 = None

Token化词表 = GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")
Token化Bert词表 = BertTokenizer.from_pretrained("Bert_Token/", pad_token="[PAD]")

# ##################################################################### 命令行参数 ######################################
def 接收命令行参数函数():
    """ 设置训练参数 """
    """
    ArgumentParser 对象
    class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.
                                    HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, 
                                    conflict_handler='error', add_help=True, allow_abbrev=True)
    prog - 程序的名称（默认：sys.argv[0]）
    usage - 描述程序用途的字符串（默认值：从添加到解析器的参数生成）
    description - 在参数帮助文档之前显示的文本（默认值：无）
    epilog - 在参数帮助文档之后显示的文本（默认值：无）
    parents - 一个 ArgumentParser 对象的列表，它们的参数也应包含在内
    formatter_class - 用于自定义帮助文档输出格式的类
    prefix_chars - 可选参数的前缀字符集合（默认值：’-’）
    fromfile_prefix_chars - 当需要从文件中读取其他参数时，用于标识文件名的前缀字符集合（默认值：None）
    argument_default - 参数的全局默认值（默认值： None）
    conflict_handler - 解决冲突选项的策略（通常是不必要的）
    add_help - 为解析器添加一个 -h/--help 选项（默认值： True）
    allow_abbrev - 如果缩写是无歧义的，则允许缩写长选项 （默认值：True）
    add_argument() 方法
    ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    """
    # ----------------------------------- 命令行解析模块 -----------#
    命令行解析器 = argparse.ArgumentParser()

    """
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。  （给属性名之前加上“--”，就能将之变为可选参数）
    action - 当参数在命令行中出现时使用的动作基本类型。 action='store_true'其实是False， action='store_false'其实是True
    nargs - 命令行参数应当消耗的数目。
    const - 被一些 action 和 nargs 选择所需求的常数。
    default - 当参数未在命令行中出现时使用的值。
    type - 命令行参数应当被转换成的类型。
    choices - 可用的参数的容器。
    required - 此命令行选项是否可省略（仅选项可用）。
    help - 一个此选项作用的简单描述。
    metavar - 在使用方法消息中使用的参数值示例。
    dest - 被添加到 parse_args() 所返回对象上的属性名。
    """

    命令行解析器.add_argument('--topk', default=1, type=int, required=False, help='最高k选1，默认8')
    命令行解析器.add_argument('--topp', default=0.5, type=float, required=False, help='最高积累概率，默认0.9')  # 0

    命令行解析器.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')  # 0,1
    命令行解析器.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')

    命令行解析器.add_argument('--arch', default="EleutherAI/gpt-neo-125M", help='模型参数')

    命令行解析器.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    命令行解析器.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    命令行解析器.add_argument('--train_raw_path', default='smalldata/data.txt', type=str, required=False, help='原始训练语料')
    命令行解析器.add_argument('--train_tokenized_path', default='smalldata/train_tokenized.txt', type=str, required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    命令行解析器.add_argument('--log_path', default='log/预测日志.txt', type=str, required=False, help='预测日志存放位置')

    命令行解析器.add_argument('--epochs', default=1, type=int, required=False, help='训练的轮次')
    命令行解析器.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')  # 8
    命令行解析器.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    命令行解析器.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    命令行解析器.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    命令行解析器.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    命令行解析器.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    命令行解析器.add_argument('--已训练本地模型路径', default='已生成模型', type=str, required=False, help='已训练本地模型路径')
    命令行解析器.add_argument('--是否使用已训练本地模型', default=1, type=str, required=False, help='0为否，1为是')
    命令行解析器.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    命令行解析器.add_argument('--seed', type=int, default=666, help='设置种子用于生成随机数，以使得训练的结果是确定的')  # None
    命令行解析器.add_argument('--num_workers', type=int, default=5, help="dataloader加载数据时使用的线程数量")

    # --------------------------- 解析参数 ------------------#
    命令行参数 = 命令行解析器.parse_args()

    return 命令行参数


# ##################################################################### 随机种子函数 ####################################
def 设置随机种子函数(命令行参数):
    """
    设置训练的随机种子
    """
    torch.manual_seed(命令行参数.seed)
    random.seed(命令行参数.seed)
    np.random.seed(命令行参数.seed)

    if 命令行参数.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ##################################################################### 日志文件 ######################################
def 创建日志文件函数(命令行参数):
    """
    将日志输出到日志文件和控制台
    """
    日志 = logging.getLogger(__name__)
    日志.setLevel(logging.INFO)

    # ----------------------- 格式 ---------------------#
    时间格式 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ------------------- 创建一个handler，用于写入 日志文件 ---------------#
    日志文件写手 = logging.FileHandler(filename=命令行参数.log_path)
    日志文件写手.setFormatter(时间格式)
    日志文件写手.setLevel(logging.INFO)
    日志.addHandler(日志文件写手)

    # ------------------- 创建一个handler，用于将日志输出到 控制台（console） -----------#
    控制台 = logging.StreamHandler()
    控制台.setLevel(logging.DEBUG)
    控制台.setFormatter(时间格式)
    日志.addHandler(控制台)

    return 日志


# ##################################################################### 创建模型 ######################################
def 创建模型函数(命令行参数):

    模型 = 模型类(命令行参数)

    # --------------------------------- 根据tokenizer的vocabulary 调整 GPT2模型的voca的大小 --------#
    # 模型.resize_token_embeddings(词表词数)

    return 模型



def 补足填充函数(一批数据):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param 一批数据:
    :return:
    标题,问题描述主体,Input描述,Output描述,输入输出样例测试和Note描述, 某慢速代码, 标签代码
    """
    # ---------------------------------------------------------- 转化为张量字典 --------------------------------#
    代码张量字典 = Token化词表(一批数据[0], max_length=768, truncation=True, padding=True, return_tensors="pt")

    return 代码张量字典


# ##################################################################### 训练函数 #######################################
def 预测生成函数(模型, 设备, 数据集, 多块GPU, 命令行参数, 训练集或测试集):

    # ---------------------------- 将 训练数据 为 训练数据加载器 --------------------#
    数据加载器 = DataLoader(dataset=数据集,  # 使用的数据集
                         batch_size=命令行参数.batch_size,  # 批处理样本大小
                         shuffle=False,  # 每次迭代前不打乱数据
                         num_workers=命令行参数.num_workers,  # 使用一个进程
                         collate_fn=补足填充函数
                         )

    # -------- 进入预测状态：不计算梯度 --------#
    模型.eval()

    # --------------------------------------- 写入日志 --------------------------------#
    日志.info('############################# 开始生成：starting training ############################')


    # ############################################## 开始 ########################################## #
    # =========================== 无梯度 ======================#
    with torch.no_grad():
        # ---------------------------------- 计算Bleu ----------------------------------------#
        运行时间总列表 = []

        # ------------------- 数据加载器循环迭代 -------------#
        # ---------- 特征列表：标题,问题描述主体,Input描述,Output描述,输入输出样例测试和Note描述, 某慢速代码, 标签代码 -------#
        for 批量索引, 代码张量字典 in enumerate(数据加载器):
            if 训练集或测试集 == "测试集" and 批量索引 % (命令行参数.log_step/10) != 0:
                continue
            else:
                # ----------------------------- 移动到GPU -------------------------------#
                代码张量字典["input_ids"] = 代码张量字典["input_ids"].to(设备)
                代码张量字典["attention_mask"] = 代码张量字典["attention_mask"].to(设备)
                try:
                    代码张量字典["token_type_ids"] = 代码张量字典["token_type_ids"].to(设备)
                except:
                    pass

                # -------------------------------- 传入GPT-2模型 ----------------#
                模型输出 = 模型(代码张量字典)

                预测的运行时间 = 模型输出.cpu().numpy().tolist()
                预测的运行时间 = 预测的运行时间[0][0]

                print(type(预测的运行时间))

                运行时间总列表.append(预测的运行时间)
                日志.info(f'\033[0:34m======================= 批次：{批量索引}. 模型输出：{预测的运行时间}   ========\033[m')


    平均运行时间 = mean(运行时间总列表)

    with open(f"预测时间.txt", 'w', encoding='UTF-8') as f:
        f.write(str(运行时间总列表))

    # ------------------------------------------ 结束：写入日志 ------------------------------------------------#
    日志.info(f'\033[0:34m======================= 平均运行时间：{平均运行时间}    结束 ========\033[m')

    return 平均运行时间


# ####################################################################### 主函数 #######################################
def main():
    # -------------------------- 初始化命令行参数 -----#
    命令行参数 = 接收命令行参数函数()

    # -------------------------------- 创建日志对象 -----#
    # ----------- 日志同时输出到文件和console -------#
    global 日志
    日志 = 创建日志文件函数(命令行参数)

    # ================================================================ 当用户使用GPU,并且GPU可用时 ===================#
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    命令行参数.cuda = torch.cuda.is_available() and not 命令行参数.no_cuda
    设备 = 'cuda' if 命令行参数.cuda else 'cpu'
    # 设备 = 'cpu'
    日志.info('using 设备:{}'.format(设备))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if 命令行参数.seed:
        设置随机种子函数(命令行参数)

    # -------------------------------------------------------- 初始化tokenizer --------------#

    # -------------------- 创建模型的输出目录 -------------------#
    if not os.path.exists(命令行参数.已训练本地模型路径):
        os.mkdir(命令行参数.已训练本地模型路径)

    # --------------------- 加载GPT模型 ------------------------#
    模型 = 创建模型函数(命令行参数)

    # ------------- 移到GPU ----#
    模型.to(设备)

    # -------------------------- 是否使用多块GPU进行并行运算 ------------------------#
    多块GPU = False
    if 命令行参数.cuda and torch.cuda.device_count() > 1:
        日志.info("Let's use 多个 GPUs to 训练")
        模型 = DataParallel(模型, device_ids=[int(i) for i in 命令行参数.device.split(',')])

    # ------------------------- 记录模型参数数量 ---------------#
    参数总数 = 0
    模型参数列表 = 模型.parameters()
    for 某层参数 in 模型参数列表:
        参数总数 += 某层参数.numel()
    # ------------------------------------ log ----------------#
    日志.info(f'=========== number of 模型 模型参数列表: {参数总数} =================')

    # -------------------------------- log ---------#
    日志.info("=================== 加载训练数据。。。 =======================")

    # ============================== 加载数据 ===============================#
    预测数据 = 训练集Dataset类(
        模型=命令行参数.arch,
        最大Token数=2048
    )

    # ----------------------------- 开始训练 ----------------------------#
    训练集BLEU分数 = 预测生成函数(模型, 设备, 预测数据, 多块GPU, 命令行参数,"训练集")

    # ---------------------------------------------------- 最终输出 ---------------------------------------------------#
    日志.info(f'\033[0:34m======= 训练集BLEU分数：{训练集BLEU分数}  ===============\033[m')

# ############################################################ 入口 ####################################################
if __name__ == '__main__':
    main()

class 预测生成代码类():
    def 入口(self):
        main()
