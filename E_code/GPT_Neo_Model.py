from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
import torch

class GPT_Neo_Model(nn.Module):

    def __init__(self, Command_line_parameters):
        super(GPT_Neo_Model, self).__init__()
        if Command_line_parameters.Whether_to_use_trained_local_models:
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        else:
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(Command_line_parameters.GPT_arch)

    def forward(self, List_of_features):
        (Question_text, slow_code, The_tag_code) = List_of_features
        Label_tensor = The_tag_code['input_ids'].clone().detach()
        for i in range(len(Label_tensor)):
            for j in range(len(Label_tensor[i])):
                if (Label_tensor[(i, j)] == 0):
                    Label_tensor[(i, j)] = (- 100)
        try:
            Final_output = self.Final_output_layer(**Question_text, labels=Label_tensor)
        except:
            Final_output = self.Final_output_layer(**Question_text)
        return Final_output