# coding=utf-8

import os

from numpy import mean

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import wandb
# wandb.login(key="82cfc67dbc70c15d489f7eaaae135311f3716c1a")

import transformers
import torch
import json
import random
import numpy as np
import argparse

from torch import tensor
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

from model import Modle
from train_Dataset import 训练集Dataset类
from test_Dataset import 测试集Dataset类
from Predictive_generation_code import 预测生成代码类

# 网址：https://www.bilibili.com/video/BV1La4y1Y7ug?spm_id_from=333.337.search-card.all.click

日志 = None

Token化词表 = transformers.GPT2Tokenizer.from_pretrained("GPT_Token/", pad_token="[PAD]", cls_token="[CLS]")
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

    命令行解析器.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')  # 0,1
    命令行解析器.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')

    命令行解析器.add_argument('--arch', default="EleutherAI/gpt-neo-125M", help='模型参数')

    命令行解析器.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    命令行解析器.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    命令行解析器.add_argument('--train_raw_path', default='smalldata/data.txt', type=str, required=False, help='原始训练语料')
    命令行解析器.add_argument('--train_tokenized_path', default='smalldata/train_tokenized.txt', type=str, required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    命令行解析器.add_argument('--log_path', default='log/训练日志.txt', type=str, required=False, help='训练日志存放位置')

    命令行解析器.add_argument('--epochs', default=30, type=int, required=False, help='训练的轮次')
    命令行解析器.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')  # 8
    命令行解析器.add_argument('--gradient_accumulation', default=32, type=int, required=False, help='梯度积累')

    命令行解析器.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    命令行解析器.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    命令行解析器.add_argument('--log_step', default=1000, type=int, required=False, help='多少步汇报一次loss')
    命令行解析器.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    命令行解析器.add_argument('--已训练本地模型路径', default='Generated_models', type=str, required=False, help='已训练本地模型路径')
    命令行解析器.add_argument('--是否使用已训练本地模型', default=0, type=str, required=False, help='0为否，1为是')
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

    模型 = Modle(命令行参数)

    # --------------------------------- 根据tokenizer的vocabulary 调整 GPT2模型的voca的大小 --------#
    # 模型.resize_token_embeddings(词表词数)

    return 模型

def 计算损失和准确性函数(模型输出, 标签, 设备):
    """
    计算非pad_id的平均loss和准确率
    :param 模型输出:
    :param 标签:
    :param 设备:
    :return:
    """
    # ============================= 每个token 用来预测下一个token的prediction_score =====================================#
    # --------------- 模型输出有两个元素，其中第一个是预测分数，其维度:[batch_size,token_len,voca_size] -----------#
    预测分数 = 模型输出[0]

    # ========================================= 用 前 n-1 个 token，预测出 第n个 token =================================#
    # ---------------------------- 用 第i个 token 的 prediction_score 用来 预测 第i+1个 token ------------------------#
    # 假定有 input 有n个token，则 移动的预测分数 表示 模型 中第[0,n-2]个token的prediction_score，
    移动的预测分数 = 预测分数[..., :-1, :].contiguous()  # .contiguous() 拷贝到连续的内存空间

    # ------------------------- 移动的标签 表示 第[1，n-1]的 label ---------------#
    移动的标签 = 标签[..., 1:].contiguous().to(设备)

    # ------------------------------------------- 交叉熵损失 -----------------------------#
    # ------------------------------ 忽略pad_id的loss,并对所有的非pad_id的loss进行求和 ----#
    损失函数 = CrossEntropyLoss(ignore_index=-100, reduction='sum')
    损失 = 损失函数(移动的预测分数.view(-1, 移动的预测分数.size(-1)), 移动的标签.view(-1))

    # --------- 预测单词id号 表示对应的prediction_score 预测出的token在voca中的id。 维度为[batch_size,token_len] -------#
    _, 预测单词id号 = 移动的预测分数.max(dim=-1)

    # ------------------------------- 对非pad_id的token的loss进行求平均，且计算出预测的准确率 ---------------------------#
    # ------------------- 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1 ----------#
    不是填充字符的位置 = 移动的标签.ne(-100)
    不是填充字符的数量 = 不是填充字符的位置.long().sum().item()  # 计算target中的非pad_id的数量

    # --------------------------- 计算model预测正确的token的个数，排除pad的tokne -------------------#
    正确矩阵 = (移动的标签 == 预测单词id号) & 不是填充字符的位置
    预测正确个数 = 正确矩阵.float().sum()

    准确率 = 预测正确个数 / 不是填充字符的数量

    损失 = 损失 / 不是填充字符的数量

    return 损失, 准确率


def 添加开始结束Token函数(inputs):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param 一批数据:
    :return:
    """
    # =================================== 开始结束Token  =====================================#
    批量大小 = len(inputs["input_ids"])

    # ----------------------------- 前一部分： input_ids -----------------------------#
    起始Token列表 = [[102]] * 批量大小
    起始Token = tensor(起始Token列表)

    结束Token列表 = [[50256]] * 批量大小
    结束Token = tensor(结束Token列表)

    inputs["input_ids"] = torch.cat((起始Token, inputs["input_ids"], 结束Token), dim=1)

    # ----------------------------- 后一部分: attention_mask ------------------------------------#
    注意力Token列表 = [[1]] * 批量大小
    注意力Token = tensor(注意力Token列表)
    inputs["attention_mask"] = torch.cat((注意力Token, inputs["attention_mask"], 注意力Token), dim=1)

    return inputs


def 补足填充函数(一批数据):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param 一批数据:
    :return:
    标题,问题描述主体,Input描述,Output描述,输入输出样例测试,Note描述, 某慢速代码, 标签代码
    """
    总问题文本 = []
    某慢速代码 = []
    标签代码 = []

    # ------------------------------- 计算该batch中input的最大长度 --------------------------#
    for 某数据 in 一批数据:
        总问题文本.append(某数据[0])
        某慢速代码.append(某数据[1])
        标签代码.append(某数据[2])

    # ---------------------------------------------------------- 转化为张量字典 --------------------------------#
    总问题文本张量字典 = Token化Bert词表(总问题文本, max_length=2048, truncation=True, padding=True, return_tensors="pt")
    某慢速代码张量字典 = Token化词表(某慢速代码, max_length=768, truncation=True, padding=True, return_tensors="pt")
    标签代码张量字典 = Token化词表(标签代码, max_length=766, truncation=True, padding=True, return_tensors="pt")

    # =================================== 开始结束Token  =====================================#
    批量大小 = len(标签代码张量字典["input_ids"])

    # ----------------------------- 前一部分： input_ids -----------------------------#
    起始Token列表 = [[102]] * 批量大小
    起始Token = tensor(起始Token列表)

    结束Token列表 = [[50256]] * 批量大小
    结束Token = tensor(结束Token列表)

    标签代码张量字典["input_ids"] = torch.cat((起始Token, 标签代码张量字典["input_ids"], 结束Token), dim=1)

    # ----------------------------- 后一部分: attention_mask ------------------------------------#
    注意力Token列表 = [[1]] * 批量大小
    注意力Token = tensor(注意力Token列表)

    标签代码张量字典["attention_mask"] = torch.cat((注意力Token, 标签代码张量字典["attention_mask"], 注意力Token), dim=1)


    return [总问题文本张量字典, 某慢速代码张量字典, 标签代码张量字典]


# ##################################################################### 训练函数 #######################################
def train函数(模型, 设备, 训练集数据, 测试集数据, 多块GPU, 命令行参数):

    # ---------------------------- 将 训练数据 为 训练数据加载器 --------------------#
    训练数据加载器 = DataLoader(dataset=训练集数据,  # 使用的数据集
                         batch_size=命令行参数.batch_size,  # 批处理样本大小
                         shuffle=True,  # 每次迭代前打乱数据
                         num_workers=命令行参数.num_workers,  # 使用一个进程
                         collate_fn=补足填充函数
                         )

    # ------------- 进入训练状态 --------#
    模型.train()

    # ------------------------------- 计算所有epoch进行参数优化的总步数total_steps ---------------------------#
    总步数 = int(训练数据加载器.__len__() * 命令行参数.epochs / 命令行参数.batch_size / 命令行参数.gradient_accumulation)

    # ------------------------- 写入日志 ------------------------#
    日志.info(f'-------- 总步数：total training steps = {总步数} ---------')

    # ---------------------- 设置优化器，并且在初始训练时，使用 warm up 策略 ---------------------#
    优化器 = transformers.AdamW(模型.parameters(), lr=命令行参数.lr, correct_bias=True)
    # scheduler = transformers.WarmupLinearSchedule(优化器, warmup_steps=命令行参数.warmup_steps, t_total=总步数)

    # --------------------------------------- 写入日志 --------------------------------#
    日志.info('####################### 开始训练：starting training ###################')

    # ------------------ 用于统计每次梯度累计的loss -------#
    每次梯度累计的损失 = 0

    # ------------------- 记录tensorboardX ------------------#
    # 可视化tbX = SummaryWriter(log_dir=命令行参数.writer_dir)

    # ############################################## 开始训练 ################################### #
    使用时间列表 = []
    for 某轮 in range(命令行参数.epochs):

        # --------- 开始时间 ------------#
        总使用时间 = 0

        # ----------- 记录 out of memory的次数 -------#
        GPU内存不足次数 = 0

        # ------------------- 数据加载器循环迭代 -------------#
        训练损失列表 = []
        # ---------- 特征列表：标题,问题描述主体,Input描述,Output描述,输入输出样例测试,Note描述, 某慢速代码, 标签代码 -------#
        for 批量索引, 特征列表 in enumerate(训练数据加载器):

            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token

            # ----------------------------- 移动到GPU -------------------------------#
            for 索引, 某张量字典 in enumerate(特征列表):
                特征列表[索引]["input_ids"] = 某张量字典["input_ids"].to(设备)
                特征列表[索引]["attention_mask"] = 某张量字典["attention_mask"].to(设备)
                try:
                    特征列表[索引]["token_type_ids"] = 某张量字典["token_type_ids"].to(设备)
                except:
                    pass

            # --------------- 解决在运行过程中，由于显存不足产生的cuda out of memory的问题 ---------------#
            try:    # https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
                # 模型输出 = 模型.forward(input_ids=特征)
                模型输出,使用时间 = 模型(特征列表)
                总使用时间 = 总使用时间 + 使用时间
                损失 = 模型输出.loss
                预测矩阵 = 模型输出.logits

                # ---------------------------- 累积损失 ------------------------#
                训练损失列表.append(损失.item())

                # ------------------------------- 更新日志与tnesorboardX信息 --------------------------------------#
                if 批量索引 % 命令行参数.log_step == 0:
                    日志.info(f'\033[0:32m第 {某轮+1} 轮(共{命令行参数.epochs}轮)，第 {批量索引+1} 批数据(共{len(训练数据加载器)}批), 损失： {损失:.3f}\033[m')
                    # --------------------------- 可视化界面 --------------#
                    # 可视化tbX.add_scalar('损失：', 损失.item(), 总训练步数)


                # ------------------------------ 特征标签相同（错了一步位置） ----------------------#
                # 损失, 准确率 = 计算损失和准确性函数(模型输出, 标签=特征列表[-1], 设备=设备)

                # -------------- 如果是多个GPU，均值 -------------#
                if 多块GPU:
                    损失 = 损失.mean()

                # -------------- 如果是梯度累积，相除 -------------#
                if 命令行参数.gradient_accumulation > 1:
                    损失 = 损失 / 命令行参数.gradient_accumulation

                # -------------- 梯度反向传播 --------#
                损失.backward()

                # ---------------------- 梯度裁剪：解决的是梯度消失或爆炸的问题，即设定阈值 -----------------#
                torch.nn.utils.clip_grad_norm_(模型.parameters(), 命令行参数.max_grad_norm)

                # ------------------ 进行一定step的梯度累计之后，更新参数 --------------#
                if (批量索引 + 1) % 命令行参数.gradient_accumulation == 0:
                    每次梯度累计的损失 += 损失.item()
                    # ---------------- 更新参数 -------#
                    优化器.step()
                    # ---------------- 清空梯度信息 -----#
                    优化器.zero_grad()
                    # ---------------- 进行warm up -----#
                    # scheduler.step()

            # ============================= 解决在运行过程中，由于显存不足产生的cuda out of memory的问题 ===================#
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    GPU内存不足次数 += 1
                    日志.info(f"#======== WARNING: GPU内存不足次数：ran out of memory,times: {GPU内存不足次数} ========#")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    日志.info(str(exception))
                    raise exception

        # ========================== 某轮训练结束 ================#
        使用时间列表.append(总使用时间)

        # ======================================================== 每结束一轮，保存模型一次 ============================#
        模型.打通的专家组层.save_pretrained("./Generated_models/打通的专家组层/")
        模型.专家组整合层.save_pretrained("./Generated_models/专家组整合层/")
        torch.save(模型.全连接放大层, "Generated_models/全连接放大层.pkl")
        模型.低效代码层.save_pretrained("./Generated_models/低效代码层/")
        模型.高效代码层.save_pretrained("./Generated_models/高效代码层/")
        torch.save(模型.编码器解码器自注意力层, "Generated_models/编码器解码器自注意力层.pkl")
        模型.最终输出层.save_pretrained("./Generated_models/最终输出层/")

        # ======================================================== 奇数轮备用：间隔保存，安全保障 ==================#
        if (某轮 + 1) % 2 == 1:
            模型.打通的专家组层.save_pretrained("./Generated_models/打通的专家组层/")
            模型.专家组整合层.save_pretrained("./奇数轮备用/专家组整合层/")
            torch.save(模型.全连接放大层, "./奇数轮备用/全连接放大层.pkl")
            模型.低效代码层.save_pretrained("./奇数轮备用/低效代码层/")
            模型.高效代码层.save_pretrained("./奇数轮备用/高效代码层/")
            torch.save(模型.编码器解码器自注意力层, "./奇数轮备用/编码器解码器自注意力层.pkl")
            模型.最终输出层.save_pretrained("./奇数轮备用/最终输出层/")
        # ======================================================== 20备用：间隔保存，安全保障 ==================#
        if (某轮 + 1)  == 20:
            模型.打通的专家组层.save_pretrained("./Generated_models/打通的专家组层/")
            模型.专家组整合层.save_pretrained("./第20轮保存/专家组整合层/")
            torch.save(模型.全连接放大层, "./第20轮保存/全连接放大层.pkl")
            模型.低效代码层.save_pretrained("./第20轮保存/低效代码层/")
            模型.高效代码层.save_pretrained("./第20轮保存/高效代码层/")
            torch.save(模型.编码器解码器自注意力层, "./第20轮保存/编码器解码器自注意力层.pkl")
            模型.最终输出层.save_pretrained("./第20轮保存/最终输出层/")
        # ======================================================== 25备用：间隔保存，安全保障 ==================#
        if (某轮 + 1) == 25:
            模型.打通的专家组层.save_pretrained("./Generated_models/打通的专家组层/")
            模型.专家组整合层.save_pretrained("./第25轮保存/专家组整合层/")
            torch.save(模型.全连接放大层, "./第25轮保存/全连接放大层.pkl")
            模型.低效代码层.save_pretrained("./第25轮保存/低效代码层/")
            模型.高效代码层.save_pretrained("./第25轮保存/高效代码层/")
            torch.save(模型.编码器解码器自注意力层, "./第25轮保存/编码器解码器自注意力层.pkl")
            模型.最终输出层.save_pretrained("./第25轮保存/最终输出层/")

        # ----------------------------------------------- 写入日志 ----------------------------------------------------#
        日志.info(f'\033[0:34m===== 训练集,第 {某轮+1} 轮结束，平均损失：{mean(训练损失列表)}。模型已经保存。总运行时间：{使用时间列表} =======\033[m')

        # ------------------------------- 测试模型 ------------------#
        测试平均损失 = 预测函数(模型, 设备, 测试集数据, 多块GPU, 命令行参数)

    # ------------------------------------------ 训练结束：写入日志 ------------------------------------------------#
    # log.info(f'\033[0:34m======================= 结束 training finished测试平均损失：{测试平均损失} ============\033[m')
    日志.info(f'\033[0:35m################## 结束。训练平均损失：{mean(训练损失列表)}。测试平均损失：{测试平均损失} ###################\033[m')


# ##################################################################### 预测函数 #######################################
def 预测函数(模型, 设备, 测试集数据, 多块GPU, 命令行参数):
    # -------------------------------------- 开始预测：写入日志 -----------------------------#
    日志.info("============================== 开始预测模型 =================================")

    # -------- 进入预测状态：不计算梯度 --------#
    模型.eval()

    # ----------------------------------- 将 测试数据 为 测试数据加载器 --------------------#
    测试数据加载器 = DataLoader(测试集数据,  # 使用的数据集
                         batch_size=命令行参数.batch_size,  # 批处理样本大小
                         shuffle=True,  # 每次迭代前打乱数据
                         num_workers=命令行参数.num_workers,  # 使用一个进程
                         collate_fn=补足填充函数
                         )

    # =========================== 无梯度 ======================#
    with torch.no_grad():
        损失列表 = []
        # --------- 开始时间 ------------#
        总使用时间 = 0
        for 批量索引, 特征列表 in enumerate(测试数据加载器):

            # ----------------------------- 移动到GPU -------------------------------#
            for 索引, 某张量字典 in enumerate(特征列表):
                特征列表[索引]["input_ids"] = 某张量字典["input_ids"].to(设备)
                特征列表[索引]["attention_mask"] = 某张量字典["attention_mask"].to(设备)
                try:
                    特征列表[索引]["token_type_ids"] = 某张量字典["token_type_ids"].to(设备)
                except:
                    pass

            # ----------------------------------------------#
            try:
                模型输出,使用时间 = 模型(特征列表)
            except:
                continue
            总使用时间 = 总使用时间 + 使用时间
            损失 = 模型输出.loss
            预测矩阵 = 模型输出.logits

            # ---------------------------- 累积损失 ------------------------#
            损失列表.append(损失.item())

            # ------------------------------ 特征标签相同（错了一步位置） ----------------------#
            # 损失, 准确率 = 计算损失和准确性函数(模型输出, 标签=特征, 设备=设备)

            # -------------- 如果是多个GPU，均值 -------------#
            if 多块GPU:
                损失 = 损失.mean()

            # -------------- 如果是梯度累积，相除 -------------#
            if 命令行参数.gradient_accumulation > 1:
                损失 = 损失 / 命令行参数.gradient_accumulation


        # -------------------------------- 预测结果：写入日志 -------------------------------------------------#
        日志.info(f'\033[0:32m预测模式:  损失： {mean(损失列表):.3f}， 总使用时间： {总使用时间}\033[m')

        # --------------------------------------- 结束预测：写入日志 --------------------------------------#
        日志.info('\033[0:34m========================== 结束预测 ==================================\033[m')

        return mean(损失列表)


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
    日志.info(f'=========== 模型总参数: {参数总数} =================')

    # -------------------------------- log ---------------------------------#
    日志.info("=================== 加载训练数据。。。 =======================")

    # ============================== 加载数据 ===============================#
    训练数据 = 训练集Dataset类(
        模型=命令行参数.arch,
        最大Token数=2048
    )
    测试数据 = 测试集Dataset类(
        模型=命令行参数.arch,
        最大Token数=2048
    )

    # ----------------------------- 开始训练 ------------------------#
    train函数(模型, 设备, 训练数据,测试数据, 多块GPU, 命令行参数)

    # --------------------------- 预测代码生成 ------------------------#
    预测生成代码 = 预测生成代码类()
    预测生成代码.入口()

# ############################################################ 入口 ####################################################
if __name__ == '__main__':
    main()
