# coding=utf-8


from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


from torch.utils.data import Dataset
import torch


# ######################################################## 搭建 全连接神经网络回归网络 #################################
class Modle(nn.Module):
    # =========================================== 初始化 =======================#
    def __init__(self,args):
        super(Modle, self).__init__()

        # ============================ 如果有已经训练的本地model， 则使用已经训练的本地model ===========================#
        if args.Whether_to_use_trained_local_mods:
            print(f'\033[0:34m使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            # ---------------------------------------- Final_output_layer ----------------------------------------#
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{args.Generated_models}/Final_output_layer/")

        # ================================ 如果没有已经训练的 使用 初始model ================================#
        else:
            print(f'\033[0:34m 不 使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            # ---------------------------------------- Final_output_layer ----------------------------------------#
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    # ################################################## 定义网络的向前传播路径 #########################################
    def forward(self, Feature_List):
        # ------------------------------- 分解 ------------------------------#
        Question_text, certain_slow_code, label_code = Feature_List

        # ---------------------------------------- label_code处理 ----------------------------------------#
        Label_Tensor = label_code["input_ids"].clone().detach()
        for i in range(len(Label_Tensor)):
            for j in range(len(Label_Tensor[i])):
                if Label_Tensor[i, j] == 0:
                    Label_Tensor[i, j] = -100

        # ----------------------------------- 特殊处理 --------------------------------#


        # ---------------------------------------- Final_output_layer ----------------------------------------#
        try:
            Final_Output = self.Final_output_layer(**Question_text, labels=Label_Tensor)
        except :
            Final_Output = self.Final_output_layer(**Question_text)


        # ----------------------------------- 输出 -------------------
        return Final_Output



