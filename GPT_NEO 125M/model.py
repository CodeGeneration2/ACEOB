

from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


from torch.utils.data import Dataset
import torch


class Modle(nn.Module):
    def __init__(self,args):
        super(Modle, self).__init__()

        if args.Whether_to_use_trained_local_mods:
            print(f'\033[0:34mmodel  args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{args.Generated_models}/Final_output_layer/")

        else:
            print(f'\033[0:34m model args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    def forward(self, Feature_List):
        Question_text, certain_slow_code, label_code = Feature_List

        Label_Tensor = label_code["input_ids"].clone().detach()
        for i in range(len(Label_Tensor)):
            for j in range(len(Label_Tensor[i])):
                if Label_Tensor[i, j] == 0:
                    Label_Tensor[i, j] = -100



        try:
            Final_Output = self.Final_output_layer(**Question_text, labels=Label_Tensor)
        except :
            Final_Output = self.Final_output_layer(**Question_text)


        return Final_Output



