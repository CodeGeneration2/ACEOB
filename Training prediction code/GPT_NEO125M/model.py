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
    def __init__(self,命令行参数):
        super(Modle, self).__init__()

        # ============================ 如果有已经训练的本地模型， 则使用已经训练的本地模型 ===========================#
        if 命令行参数.是否使用已训练本地模型:
            print(f'\033[0:34m使用已训练本地模型，命令行参数.是否使用已训练本地模型：{命令行参数.是否使用已训练本地模型} \033[m')

            # ---------------------------------------- 最终输出层 ----------------------------------------#
            self.最终输出层 = transformers.GPTNeoForCausalLM.from_pretrained(f"{命令行参数.已训练本地模型路径}/最终输出层/")

        # ================================ 如果没有已经训练的 使用 初始模型 ================================#
        else:
            print(f'\033[0:34m 不 使用已训练本地模型，命令行参数.是否使用已训练本地模型：{命令行参数.是否使用已训练本地模型} \033[m')

            # ---------------------------------------- 最终输出层 ----------------------------------------#
            self.最终输出层 = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    # ################################################## 定义网络的向前传播路径 #########################################
    def forward(self, 特征列表):
        # ------------------------------- 分解 ------------------------------#
        问题文本, 某慢速代码, 标签代码 = 特征列表

        # ---------------------------------------- 标签代码处理 ----------------------------------------#
        标签张量 = 标签代码["input_ids"].clone().detach()
        for i in range(len(标签张量)):
            for j in range(len(标签张量[i])):
                if 标签张量[i, j] == 0:
                    标签张量[i, j] = -100

        # ----------------------------------- 特殊处理 --------------------------------#


        # ---------------------------------------- 最终输出层 ----------------------------------------#
        最终输出 = self.最终输出层(**问题文本, labels=标签张量)

        # ----------------------------------- 输出 -------------------
        return 最终输出


if __name__ == '__main__':
    """
    # tokenizer1 = BertTokenizer.from_pretrained("Bert_small_权重/vocab.txt")
    tokenizer = BertTokenizer.from_pretrained("Bert_small_权重/")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_json_file("Bert_small_权重/config.json")
    model = BertModel.from_pretrained("Bert_small_权重/", config=config)

    print(model)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state



    last_hidden_states = 33321
"""

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

    tokenizer = BertTokenizer.from_pretrained("Bert_small_权重/")
    model = Modle()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=[inputs, inputs],

        tokenizer=tokenizer,

    )

    # tokenizer1 = BertTokenizer.from_pretrained("Bert_small_权重/vocab.txt")

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    trainer.train()

    print("wd")






