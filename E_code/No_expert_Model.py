from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
import torch

class No_expert_Model(nn.Module):

    def __init__(self, Command_line_parameters):
        super(No_expert_Model, self).__init__()
        if Command_line_parameters.Whether_to_use_trained_local_models:
            self.Opened_expert_group_layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Opened_expert_group_layer/')
            self.Expert_Group_Integration_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Expert_Group_Integration_Layer/')
            self.MLP_enlarge_layer = torch.load(f'{Command_line_parameters.Generated_models}/MLP_enlarge_layer.pkl')
            self.IC_layer = transformers.GPTNeoModel.from_pretrained(f'{Command_line_parameters.Generated_models}/IC_layer/')
            self.EC_layer = transformers.GPTNeoModel.from_pretrained(f'{Command_line_parameters.Generated_models}/EC_layer/')
            self.Multi_headed_attention_mechanism = torch.load(f'{Command_line_parameters.Generated_models}/Multi_headed_attention_mechanism.pkl')
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        else:
            Bert_parameters = BertConfig.from_json_file('Bert_tiny_Opened_expert_group_layer_Weights/config.json')
            self.Opened_expert_group_layer = BertModel(config=Bert_parameters)
            Bert_parameters = BertConfig.from_json_file('Weights_Expert_Group_Integration_Layer/config.json')
            self.Expert_Group_Integration_Layer = BertModel(config=Bert_parameters)
            if Command_line_parameters.RELU:
                self.MLP_enlarge_layer = nn.Sequential(nn.Linear(in_features=128, out_features=768, bias=True), nn.ReLU())
            else:
                self.MLP_enlarge_layer = nn.Linear(in_features=128, out_features=768, bias=True)
            self.IC_layer = transformers.GPTNeoModel.from_pretrained(Command_line_parameters.GPT_arch)
            self.EC_layer = transformers.GPTNeoModel.from_pretrained(Command_line_parameters.GPT_arch)
            self.Multi_headed_attention_mechanism = nn.MultiheadAttention(embed_dim=768, num_heads=Command_line_parameters.heads, batch_first=True)
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(Command_line_parameters.GPT_arch)

    def forward(self, List_of_features):
        (Total_problem_text, slow_code, The_tag_code) = List_of_features
        Opened_expert_group_layer_output = self.Opened_expert_group_layer(**Total_problem_text).last_hidden_state
        Expert_group_output = self.Expert_Group_Integration_Layer(inputs_embeds=Opened_expert_group_layer_output).last_hidden_state
        Text_layer_output = self.MLP_enlarge_layer(Expert_group_output)
        Inefficient_Code_Layer_Output = self.IC_layer(**slow_code).last_hidden_state
        Encoder_output = torch.cat((Text_layer_output, Inefficient_Code_Layer_Output), dim=1)
        Efficient_code_layer_output = self.EC_layer(**The_tag_code).last_hidden_state
        (Encoder_Decoder_Self_Attention_Layer_Output, attn_output_weights) = self.Multi_headed_attention_mechanism(Efficient_code_layer_output, Encoder_output, Encoder_output)
        Label_tensor = The_tag_code['input_ids'].clone().detach()
        for i in range(len(Label_tensor)):
            for j in range(len(Label_tensor[i])):
                if (Label_tensor[(i, j)] == 0):
                    Label_tensor[(i, j)] = (- 100)
        Final_output = self.Final_output_layer(inputs_embeds=Encoder_Decoder_Self_Attention_Layer_Output, labels=Label_tensor)
        return Final_output