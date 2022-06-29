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
import time


# ######################################################## 搭建 全连接神经网络回归网络 #################################
class Modle(nn.Module):
    # =========================================== 初始化 =======================#
    def __init__(self,args):
        super(Modle, self).__init__()

        # ============================ 如果有已经训练的本地model， 则使用已经训练的本地model ===========================#
        if args.Whether_to_use_trained_local_mods:
            print(f'\033[0:34m使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')
            """
            # ---------------- 定义专家组:bert输出矩阵【批量尺寸，句子长度，词嵌入】 -----------#
            bert-tiny: 词嵌入维度128，最大长度512
            GPT-NEO125M: 词嵌入维度768，最大长度2048
            
            self.title_layer = BertModel.from_pretrained(f"{args.Generated_models}/title_layer/")
            self.problem_description_subject_layer = BertModel.from_pretrained(f"{args.Generated_models}/problem_description_subject_layer/")
            self.Input_description_layer = BertModel.from_pretrained(f"{args.Generated_models}/Input_description_layer/")
            self.Output_Description_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Output_Description_Layer/")
            self.Input_output_sample_tests_and_Note_description_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Input_output_sample_tests_and_Note_description_Layer/")
            """
            # ------------------------------- model.Opened_expert_group_layer ------------------------------------#
            self.Opened_expert_group_layer = BertModel.from_pretrained(f"{args.Generated_models}/Opened_expert_group_layer/")

            # ------------------------------- Expert_Group_Integration_Layer ------------------------------------#
            self.Expert_Group_Integration_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Expert_Group_Integration_Layer/")

            # ---------------------------------------- Fully_connected_amplification_layer -----------------------#
            self.Fully_connected_amplification_layer = torch.load(f"{args.Generated_models}/Fully_connected_amplification_layer.pkl")

            # ---------------------------------------- Inefficient_code_layer ----------------------------------------
            """代码层：使用GPT——NEOmodel
            """
            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Inefficient_code_layer/")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Efficient_code_layer/")

            # ---------------------------------------- Encoder_Decoder_Self_Attention_Layer ----------------------------------------#
            """ 自注意力层：（疑问测试）
            1，Q K V :   分开
            2,8个头改动
            3，是否加feedforward，全连接层
            """
            self.Encoder_Decoder_Self_Attention_Layer = torch.load(f"{args.Generated_models}/Encoder_Decoder_Self_Attention_Layer.pkl")

            # ---------------------------------------- Final_output_layer ----------------------------------------#
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{args.Generated_models}/Final_output_layer/")

        # ================================ 如果没有已经训练的 使用 初始model ================================#
        else:
            print(f'\033[0:34m 不 使用已训练本地model，args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            # --------------------- 定义专家组参数 --------------------------------#
            Bert_parameters = BertConfig.from_json_file("Bert_tiny_Get_hrough_expert_group_weights/config.json")
            self.Opened_expert_group_layer = BertModel(config=Bert_parameters)

            # ---------------- 定义专家组:bert输出矩阵【批量尺寸，句子长度，词嵌入】 -----------#
            # bert-tiny: 词嵌入维度128，最大长度512
            # GPT-NEO125M: 词嵌入维度768，最大长度2048
            """
            self.title_layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert_parameters)
            self.problem_description_subject_layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert_parameters)
            self.Input_description_layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert_parameters)
            self.Output_Description_Layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert_parameters)
            self.Input_output_sample_tests_and_Note_description_Layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert_parameters)
            """
            # ------------------------------- Expert_Group_Integration_Layer ------------------------------------#
            """ 由于bert-tiny的最大长度是512，所以这里无法使用已经预训练的bert-tiny，所以从头开始训练"""
            Bert_parameters = BertConfig.from_json_file("Bert_tiny_weight_Expert_group_integration_layer/config.json")
            self.Expert_Group_Integration_Layer = BertModel(config=Bert_parameters)

            # ---------------------------------------- Fully_connected_amplification_layer -----------------------#
            self.Fully_connected_amplification_layer = nn.Linear(in_features=128, out_features=768, bias=True)

            # ---------------------------------------- Inefficient_code_layer ----------------------------------------
            """代码层：使用GPT——NEOmodel
            """
            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")

            # ---------------------------------------- Encoder_Decoder_Self_Attention_Layer ----------------------------------------#
            """ 自注意力层：（疑问测试）
            1，Q K V :   分开
            2,8个头改动
            3，是否加feedforward，全连接层  
            """
            self.Encoder_Decoder_Self_Attention_Layer = nn.MultiheadAttention(embed_dim=768, num_heads=48,batch_first=True)

            # ---------------------------------------- Final_output_layer ----------------------------------------#
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    # ################################################## 定义网络的向前传播路径 #########################################
    def forward(self, Feature_List):
        # ------------------------------- 分解 ------------------------------#
        General_question_text, certain_slow_code, label_code = Feature_List


        # ------------------------------- 定义专家组 ------------------------------#
        Opened_expert_group_layer_output = self.Opened_expert_group_layer(**General_question_text).last_hidden_state

        # ---------------------------------------- Expert_Group_Integration_Layer -------------#
        """输入：张量 [batch_size，序列长度2048，词嵌入维度128]
        输出：[batch_size，序列长度2048，词嵌入维度128]
        """
        Expert_Group_Output = self.Expert_Group_Integration_Layer(inputs_embeds=Opened_expert_group_layer_output).last_hidden_state

        """           ---------------------- 可以后放大，也可以先放大  ------------- 试试效果 ----------------"""

        # ---------------------------------------- Fully_connected_amplification_layer -------------#
        """输入：张量 [batch_size，序列长度2048，词嵌入维度128]
        输出：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        Text_Layer_Output = self.Fully_connected_amplification_layer(Expert_Group_Output)

        # ---------------------------------------- Inefficient_code_layer -------------#
        """输入：{input_ids：张量，attention_mask：张量}
        输出：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        Inefficient_code_layer_output = self.Inefficient_code_layer(**certain_slow_code).last_hidden_state

        # ---------------------------------------- Encoder_output -------------#
        Encoder_output = torch.cat((Text_Layer_Output, Inefficient_code_layer_output), dim=1)

        # -------------------------------------------- Efficient_code_layer ----------------------------------------------------#
        """输入：{input_ids：张量，attention_mask：张量}
        输出：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        Efficient_code_layer_output = self.Efficient_code_layer(**label_code).last_hidden_state

        # -------------------------------------------- Encoder_Decoder_Self_Attention_Layer -----------------------------------------#
        """输入：QKV
        输出：张量 [batch_size，序列长度2048，词嵌入维度768]
        """
        Encoder_Decoder_Self_Attention_Layer_output, attn_output_weights = self.Encoder_Decoder_Self_Attention_Layer(Efficient_code_layer_output,Encoder_output,Encoder_output)

        # ---------------------------------------- label_code处理 ----------------------------------------#
        Label_Tensor = label_code["input_ids"].clone().detach()
        for i in range(len(Label_Tensor)):
            for j in range(len(Label_Tensor[i])):
                if Label_Tensor[i, j] == 0:
                    Label_Tensor[i, j] = -100

        # ---------------------------------------- Final_output_layer ----------------------------------------#
        Final_Output = self.Final_output_layer(inputs_embeds=Encoder_Decoder_Self_Attention_Layer_output, labels=Label_Tensor)

        # ----------------------------------- 输出 -------------------
        return Final_Output



