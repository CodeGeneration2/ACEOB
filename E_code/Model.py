from transformers import BertTokenizer, BertModel
import transformers
from transformers import BertConfig, BertModel
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
import torch

class E_code_Model(nn.Module):

    def __init__(self, Command_line_parameters):
        super(E_code_Model, self).__init__()
        if Command_line_parameters.Whether_to_use_trained_local_models:
            self.Title_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Title_Layer/')
            self.Problem_description_body_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Problem_description_body_Layer/')
            self.Input_description_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Input_description_Layer/')
            self.Output_description_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Output_description_Layer/')
            self.IO_sample_testing_and_note_description_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/IO_sample_testing_and_note_description_Layer/')
            self.Expert_Group_Integration_Layer = BertModel.from_pretrained(f'{Command_line_parameters.Generated_models}/Expert_Group_Integration_Layer/')
            self.MLP_enlarge_layer = torch.load(f'{Command_line_parameters.Generated_models}/MLP_enlarge_layer.pkl')
            self.IC_layer = transformers.GPTNeoModel.from_pretrained(f'{Command_line_parameters.Generated_models}/IC_layer/')
            self.EC_layer = transformers.GPTNeoModel.from_pretrained(f'{Command_line_parameters.Generated_models}/EC_layer/')
            self.Multi_headed_attention_mechanism = torch.load(f'{Command_line_parameters.Generated_models}/Multi_headed_attention_mechanism.pkl')
            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f'{Command_line_parameters.Generated_models}/Final_output_layer/')
        else:
            Bert_parameters = BertConfig.from_json_file('Bert_tiny_Weights/config.json')
            self.Title_Layer = BertModel.from_pretrained('Bert_tiny_Weights/', config=Bert_parameters)
            self.Problem_description_body_Layer = BertModel.from_pretrained('Bert_tiny_Weights/', config=Bert_parameters)
            self.Input_description_Layer = BertModel.from_pretrained('Bert_tiny_Weights/', config=Bert_parameters)
            self.Output_description_Layer = BertModel.from_pretrained('Bert_tiny_Weights/', config=Bert_parameters)
            self.IO_sample_testing_and_note_description_Layer = BertModel.from_pretrained('Bert_tiny_Weights/', config=Bert_parameters)
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
        (Title, Body_of_the_problem_description, Input_description, Output_description, Input_Output_Sample_Tests_and_Note_Descriptions, slow_code, The_tag_code) = List_of_features
        Header_layer_output = self.Title_Layer(**Title).last_hidden_state
        Question_description_body_layer_output = self.Problem_description_body_Layer(**Body_of_the_problem_description).last_hidden_state
        Input_description_layer_output = self.Input_description_Layer(**Input_description).last_hidden_state
        Output_description_layer_output = self.Output_description_Layer(**Output_description).last_hidden_state
        Input_Output_Sample_Test_and_Note_Description_Layer_Output = self.IO_sample_testing_and_note_description_Layer(**Input_Output_Sample_Tests_and_Note_Descriptions).last_hidden_state
        Expert_output_linkage = torch.cat((Header_layer_output, Question_description_body_layer_output, Input_description_layer_output, Output_description_layer_output, Input_Output_Sample_Test_and_Note_Description_Layer_Output), dim=1)
        if (len(Expert_output_linkage[0]) > 2048):
            Expert_output_linkage = Expert_output_linkage[:, :2048, :]
        Expert_group_output = self.Expert_Group_Integration_Layer(inputs_embeds=Expert_output_linkage).last_hidden_state
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