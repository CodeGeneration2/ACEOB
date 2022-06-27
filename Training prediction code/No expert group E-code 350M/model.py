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
    def __init__(self,命令行参数):
        super(Modle, self).__init__()

        # ============================ 如果有已经训练的本地模型， 则使用已经训练的本地模型 ===========================#
        if 命令行参数.是否使用已训练本地模型:
            print(f'\033[0:34m使用已训练本地模型，命令行参数.是否使用已训练本地模型：{命令行参数.是否使用已训练本地模型} \033[m')
            """
            # ---------------- 定义专家组:bert输出矩阵【批量尺寸，句子长度，词嵌入】 -----------#
            bert-tiny: 词嵌入维度128，最大长度512
            GPT-NEO125M: 词嵌入维度768，最大长度2048
            
            self.title_layer = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/title_layer/")
            self.问题描述主体层 = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/问题描述主体层/")
            self.Input_description_layer = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/Input_description_layer/")
            self.Output描述层 = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/Output描述层/")
            self.输入输出样例测试和Note描述层 = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/输入输出样例测试和Note描述层/")
            """
            # ------------------------------- 模型.打通的专家组层 ------------------------------------#
            self.打通的专家组层 = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/打通的专家组层/")

            # ------------------------------- 专家组整合层 ------------------------------------#
            self.专家组整合层 = BertModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/专家组整合层/")

            # ---------------------------------------- 全连接放大层 -----------------------#
            self.全连接放大层 = torch.load(f"{命令行参数.已训练本地模型路径}/全连接放大层.pkl")

            # ---------------------------------------- 低效代码层 ----------------------------------------
            """代码层：使用GPT——NEO模型
            """
            self.低效代码层 = transformers.GPTNeoModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/低效代码层/")
            self.高效代码层 = transformers.GPTNeoModel.from_pretrained(f"{命令行参数.已训练本地模型路径}/高效代码层/")

            # ---------------------------------------- 编码器解码器自注意力层 ----------------------------------------#
            """ 自注意力层：（疑问测试）
            1，Q K V :   分开
            2,8个头改动
            3，是否加feedforward，全连接层
            """
            self.编码器解码器自注意力层 = torch.load(f"{命令行参数.已训练本地模型路径}/编码器解码器自注意力层.pkl")

            # ---------------------------------------- 最终输出层 ----------------------------------------#
            self.最终输出层 = transformers.GPTNeoForCausalLM.from_pretrained(f"{命令行参数.已训练本地模型路径}/最终输出层/")

        # ================================ 如果没有已经训练的 使用 初始模型 ================================#
        else:
            print(f'\033[0:34m 不 使用已训练本地模型，命令行参数.是否使用已训练本地模型：{命令行参数.是否使用已训练本地模型} \033[m')

            # --------------------- 定义专家组参数 --------------------------------#
            Bert的参数 = BertConfig.from_json_file("Bert_tiny_Get_hrough_expert_group_weights/config.json")
            self.打通的专家组层 = BertModel(config=Bert的参数)

            # ---------------- 定义专家组:bert输出矩阵【批量尺寸，句子长度，词嵌入】 -----------#
            # bert-tiny: 词嵌入维度128，最大长度512
            # GPT-NEO125M: 词嵌入维度768，最大长度2048
            """
            self.title_layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert的参数)
            self.问题描述主体层 = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert的参数)
            self.Input_description_layer = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert的参数)
            self.Output描述层 = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert的参数)
            self.输入输出样例测试和Note描述层 = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=Bert的参数)
            """
            # ------------------------------- 专家组整合层 ------------------------------------#
            """ 由于bert-tiny的最大长度是512，所以这里无法使用已经预训练的bert-tiny，所以从头开始训练"""
            Bert的参数 = BertConfig.from_json_file("Bert_tiny_weight_Expert_group_integration_layer/config.json")
            self.专家组整合层 = BertModel(config=Bert的参数)

            # ---------------------------------------- 全连接放大层 -----------------------#
            self.全连接放大层 = nn.Linear(in_features=128, out_features=768, bias=True)

            # ---------------------------------------- 低效代码层 ----------------------------------------
            """代码层：使用GPT——NEO模型
            """
            self.低效代码层 = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            self.高效代码层 = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")

            # ---------------------------------------- 编码器解码器自注意力层 ----------------------------------------#
            """ 自注意力层：（疑问测试）
            1，Q K V :   分开
            2,8个头改动
            3，是否加feedforward，全连接层  
            """
            self.编码器解码器自注意力层 = nn.MultiheadAttention(embed_dim=768, num_heads=48,batch_first=True)

            # ---------------------------------------- 最终输出层 ----------------------------------------#
            self.最终输出层 = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    # ################################################## 定义网络的向前传播路径 #########################################
    def forward(self, 特征列表):
        # ------------------------------- 分解 ------------------------------#
        总问题文本, 某慢速代码, 标签代码 = 特征列表

        # -------------------- 开始时间 ------------#
        开始时间 = time.time()

        # ------------------------------- 定义5个专家组 ------------------------------#
        打通的专家组层输出 = self.打通的专家组层(**总问题文本).last_hidden_state
        # """输入：{input_ids：张量，attention_mask：张量}
        # 输出：last_hidden_state：张量，[批量大小，序列长度，词嵌入维度]
        # """
        # 标题层输出 = self.title_layer(**标题).last_hidden_state
        # 问题描述主体层输出 = self.问题描述主体层(**问题描述主体).last_hidden_state
        # Input描述层输出 = self.Input_description_layer(**Input描述).last_hidden_state
        # Output描述层输出 = self.Output描述层(**Output描述).last_hidden_state
        # 输入输出样例测试和Note描述层输出 = self.输入输出样例测试和Note描述层(**输入输出样例测试和Note描述).last_hidden_state
        #
        # # -------------------------------------------- 专家组整合层(1,特征列表,128) ------------------------------------------#
        # 专家输出联结 = torch.cat((标题层输出, 问题描述主体层输出, Input描述层输出, Output描述层输出 , 输入输出样例测试和Note描述层输出), dim=1)
        # if len(专家输出联结[0]) > 2048:
        #     print('\033[0:34m 模型内部 删除 过大的 : len(专家输出联结张量)>2048     033[m')
        #     专家输出联结 = 专家输出联结[:,:2048,:]

        # --------------------- 使用时间 ------------#
        使用时间 = time.time() - 开始时间

        # ---------------------------------------- 专家组整合层 -------------#
        """输入：张量 [批量大小，序列长度2048，词嵌入维度128]
        输出：[批量大小，序列长度2048，词嵌入维度128]
        """
        专家组输出 = self.专家组整合层(inputs_embeds=打通的专家组层输出).last_hidden_state

        """           ---------------------- 可以后放大，也可以先放大  ------------- 试试效果 ----------------"""

        # ---------------------------------------- 全连接放大层 -------------#
        """输入：张量 [批量大小，序列长度2048，词嵌入维度128]
        输出：张量 [批量大小，序列长度2048，词嵌入维度768]
        """
        文本层输出 = self.全连接放大层(专家组输出)

        # ---------------------------------------- 低效代码层 -------------#
        """输入：{input_ids：张量，attention_mask：张量}
        输出：张量 [批量大小，序列长度2048，词嵌入维度768]
        """
        低效代码层输出 = self.低效代码层(**某慢速代码).last_hidden_state

        # ---------------------------------------- 编码器输出 -------------#
        编码器输出 = torch.cat((文本层输出, 低效代码层输出), dim=1)

        # -------------------------------------------- 高效代码层 ----------------------------------------------------#
        """输入：{input_ids：张量，attention_mask：张量}
        输出：张量 [批量大小，序列长度2048，词嵌入维度768]
        """
        高效代码层输出 = self.高效代码层(**标签代码).last_hidden_state

        # -------------------------------------------- 编码器解码器自注意力层 -----------------------------------------#
        """输入：QKV
        输出：张量 [批量大小，序列长度2048，词嵌入维度768]
        """
        编码器解码器自注意力层输出, attn_output_weights = self.编码器解码器自注意力层(高效代码层输出,编码器输出,编码器输出)

        # ---------------------------------------- 标签代码处理 ----------------------------------------#
        标签张量 = 标签代码["input_ids"].clone().detach()
        for i in range(len(标签张量)):
            for j in range(len(标签张量[i])):
                if 标签张量[i, j] == 0:
                    标签张量[i, j] = -100

        # ---------------------------------------- 最终输出层 ----------------------------------------#
        最终输出 = self.最终输出层(inputs_embeds=编码器解码器自注意力层输出, labels=标签张量)

        # ----------------------------------- 输出 -------------------
        return 最终输出,使用时间


if __name__ == '__main__':
    模型=Modle()
    print(模型)

    """
    # tokenizer1 = BertTokenizer.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/vocab.txt")
    tokenizer = BertTokenizer.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file("Bert_tiny_Get_hrough_expert_group_weights/config.json")
    model = BertModel.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/", config=config)

    print(model)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state



    last_hidden_states = 33321


    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        fp16=True,
    )

    tokenizer = BertTokenizer.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/")
    model = Modle()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=[inputs, inputs],

        tokenizer=tokenizer,

    )

    # tokenizer1 = BertTokenizer.from_pretrained("Bert_tiny_Get_hrough_expert_group_weights/vocab.txt")

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    trainer.train()

    print("wd")
"""





