

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
    def __init__(self,args):
        super(Model, self).__init__()

        if args.Whether_to_use_trained_local_mods:
            print(f'\033[0:34,args.Whether_to_use_trained_local_mods{args.Whether_to_use_trained_local_mods} \033[m')

            self.Code_Layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Code_Layer/")
            self.Fully_connected_layer = torch.load(f"{args.Generated_models}/Fully_connected_layer.pkl")

        else:
            print(f'\033[0:34margs.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')

            self.Code_Layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            self.Fully_connected_layer = nn.Linear(in_features=768, out_features=1, bias=True)

    def forward(self, code):
        Code_Layer_Output = self.Code_Layer(**code).last_hidden_state

        output = self.Fully_connected_layer(Code_Layer_Output)


        return output[:,-1,]




