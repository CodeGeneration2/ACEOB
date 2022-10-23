from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
import torch

class Model(nn.Module):

    def __init__(self, Command_line_parameters):
        super(Model, self).__init__()
        if Command_line_parameters.Whether_to_use_trained_local_models:
            self.Code_layer = transformers.GPTNeoModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Code_layer/')
            self.MLP_layer = torch.load(f'{Command_line_parameters.Generated_models}/MLP_layer.pkl')
        else:
            self.Code_layer = transformers.GPTNeoModel.from_pretrained('EleutherAI/gpt-neo-125M')
            self.MLP_layer = nn.Linear(in_features=768, out_features=1, bias=True)

    def forward(self, Code):
        Code_level_output = self.Code_layer(**Code).last_hidden_state
        Output = self.MLP_layer(Code_level_output)
        return Output[:, (- 1)]