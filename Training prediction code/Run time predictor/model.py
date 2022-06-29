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
class Model(nn.Module):
    # =========================================== 初始化 =======================#
    def __init__(self,args):
        super(Model, self).__init__()

        # ============================ 如果有已经训练的本地model， 则使用已经训练的本地model ===========================#
        if args.Whether_to_use_trained_local_mods:
            print(f'\033[0:34m使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            # ---------------------------------------- Inefficient_code_layer ----------------------------------------
            """Code_Layer：使用GPT——NEOmodel
            """
            self.Code_Layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Code_Layer/")
            # ---------------------------------------- Fully_connected_amplification_layer -----------------------#
            self.Fully_connected_layer = torch.load(f"{args.Generated_models}/Fully_connected_layer.pkl")

        # ================================ 如果没有已经训练的 使用 初始model ================================#
        else:
            print(f'\033[0:34m 不 使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            # ---------------------------------------- Inefficient_code_layer ----------------------------------------
            """Code_Layer：使用GPT——NEOmodel
            """
            self.Code_Layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            # ---------------------------------------- Fully_connected_amplification_layer -----------------------#
            self.Fully_connected_layer = nn.Linear(in_features=768, out_features=1, bias=True)

    # ################################################## 定义网络的向前传播路径 #########################################
    def forward(self, code):
        # ---------------------------------------- Inefficient_code_layer -------------#
        """输入：{input_ids：张量，attention_mask：张量}
        output：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        Code_Layer_Output = self.Code_Layer(**code).last_hidden_state

        # ---------------------------------------- Fully_connected_amplification_layer -------------#
        """输入：张量 [batch_size，序列长度2048，词嵌入维度128]
        output：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        output = self.Fully_connected_layer(Code_Layer_Output)

        # # ---------------------------------------- Final_output_layer ----------------------------------------#
        # Final_Output = self.Final_output_layer(inputs_embeds=Encoder_Decoder_Self_Attention_Layer_output, labels=Label_Tensor)

        # ----------------------------------- output -------------------
        return output[:,-1,]




