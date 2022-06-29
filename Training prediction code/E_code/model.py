

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
            print(f'\033[0:34args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')
            self.title_layer = BertModel.from_pretrained(f"{args.Generated_models}/title_layer/")
            self.problem_description_subject_layer = BertModel.from_pretrained(f"{args.Generated_models}/problem_description_subject_layer/")
            self.Input_description_layer = BertModel.from_pretrained(f"{args.Generated_models}/Input_Description_Layer/")
            self.Output_Description_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Output_Description_Layer/")
            self.Input_output_sample_tests_and_Note_description_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Input_output_sample_tests_and_Note_description_Layer/")

            self.Expert_Group_Integration_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Expert_Group_Integration_Layer/")

            self.Fully_connected_amplification_layer = torch.load(f"{args.Generated_models}/Fully_connected_amplification_layer.pkl")

            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Inefficient_code_layer/")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Efficient_code_layer/")


            self.Encoder_Decoder_Self_Attention_Layer = torch.load(f"{args.Generated_models}/Encoder_Decoder_Self_Attention_Layer.pkl")

            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{args.Generated_models}/Final_output_layer/")

        else:
            print(f'\033[0:34args.Whether_to_use_trained_local_mods：{args.Whether_to_use_trained_local_mods} \033[m')
            Bert_parameters = BertConfig.from_json_file("Bert_tiny_weight/config.json")

            self.title_layer = BertModel.from_pretrained("Bert_tiny_weight/", config=Bert_parameters)
            self.problem_description_subject_layer = BertModel.from_pretrained("Bert_tiny_weight/", config=Bert_parameters)
            self.Input_description_layer = BertModel.from_pretrained("Bert_tiny_weight/", config=Bert_parameters)
            self.Output_Description_Layer = BertModel.from_pretrained("Bert_tiny_weight/", config=Bert_parameters)
            self.Input_output_sample_tests_and_Note_description_Layer = BertModel.from_pretrained("Bert_tiny_weight/", config=Bert_parameters)

            Bert_parameters = BertConfig.from_json_file("Bert_tiny_weight_Expert_group_integration_layer/config.json")
            self.Expert_Group_Integration_Layer = BertModel(config=Bert_parameters)

            self.Fully_connected_amplification_layer = nn.Linear(in_features=128, out_features=768, bias=True)

            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")

            self.Encoder_Decoder_Self_Attention_Layer = nn.MultiheadAttention(embed_dim=768, num_heads=48,batch_first=True)

            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    def forward(self, Feature_List):
        title, problem_Description_Subject, Input_Description, Output_Description, Input_output_sample_tests_and_Note_description, certain_slow_code, label_code = Feature_List

        title_layer_output = self.title_layer(**title).last_hidden_state
        problem_description_subject_layer_output = self.problem_description_subject_layer(**problem_Description_Subject).last_hidden_state
        Input_Description_Layer_output = self.Input_description_layer(**Input_Description).last_hidden_state
        Output_Description_Layer_output = self.Output_Description_Layer(**Output_Description).last_hidden_state
        Input_output_sample_tests_and_Note_description_Layer_output = self.Input_output_sample_tests_and_Note_description_Layer(**Input_output_sample_tests_and_Note_description).last_hidden_state

        Expert_output_linkage = torch.cat((title_layer_output, problem_description_subject_layer_output, Input_Description_Layer_output, Output_Description_Layer_output , Input_output_sample_tests_and_Note_description_Layer_output), dim=1)
        if len(Expert_output_linkage[0]) > 2048:
            print(f'\033[0:34m model{len(Expert_output_linkage[0])} len(Expert_output_linkage)>2048  033[m')
            Expert_output_linkage = Expert_output_linkage[:,:2048,:]

        Expert_Group_Output = self.Expert_Group_Integration_Layer(inputs_embeds=Expert_output_linkage).last_hidden_state
