

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

class Modle(nn.Module):

    def __init__(self,args):
        super(Modle, self).__init__()

        if args.Whether_to_use_trained_local_mods:
            self.Opened_expert_group_layer = BertModel.from_pretrained(f"{args.Generated_models}/Opened_expert_group_layer/")
            self.Expert_Group_Integration_Layer = BertModel.from_pretrained(f"{args.Generated_models}/Expert_Group_Integration_Layer/")
            self.Fully_connected_amplification_layer = torch.load(f"{args.Generated_models}/Fully_connected_amplification_layer.pkl")

            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Inefficient_code_layer/")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained(f"{args.Generated_models}/Efficient_code_layer/")


            self.Encoder_Decoder_Self_Attention_Layer = torch.load(f"{args.Generated_models}/Encoder_Decoder_Self_Attention_Layer.pkl")

            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained(f"{args.Generated_models}/Final_output_layer/")
        else:
            self.Bert_parameters = BertConfig.from_json_file("Bert_tiny_Get_hrough_expert_group_weights/config.json")
            self.Opened_expert_group_layer = BertModel(config=Bert_parameters)

            # ------------------------------- Expert_Group_Integration_Layer ------------------------------------#
            Bert_parameters = BertConfig.from_json_file("Bert_tiny_weight_Expert_group_integration_layer/config.json")
            self.Expert_Group_Integration_Layer = BertModel(config=Bert_parameters)

            # ---------------------------------------- Fully_connected_amplification_layer -----------------------#
            self.Fully_connected_amplification_layer = nn.Linear(in_features=128, out_features=768, bias=True)

            self.Inefficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
            self.Efficient_code_layer = transformers.GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")


            self.Encoder_Decoder_Self_Attention_Layer = nn.MultiheadAttention(embed_dim=768, num_heads=48, batch_first=True)


            self.Final_output_layer = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


    def forward(self, Feature_List):

            General_question_text, certain_slow_code, label_code = Feature_List

            Opened_expert_group_layer_output = self.Opened_expert_group_layer(**General_question_text).last_hidden_state

            Expert_Group_Output = self.Expert_Group_Integration_Layer(inputs_embeds=Opened_expert_group_layer_output).last_hidden_state


            Text_Layer_Output = self.Fully_connected_amplification_layer(Expert_Group_Output)

            Inefficient_code_layer_output = self.Inefficient_code_layer(**certain_slow_code).last_hidden_state

            Encoder_output = torch.cat((Text_Layer_Output, Inefficient_code_layer_output), dim=1)

            Efficient_code_layer_output = self.Efficient_code_layer(**label_code).last_hidden_state

            Encoder_Decoder_Self_Attention_Layer_output, attn_output_weights = self.Encoder_Decoder_Self_Attention_Layer(Efficient_code_layer_output,Encoder_output,Encoder_output)

            Label_Tensor = label_code["input_ids"].clone().detach()
            for i in range(len(Label_Tensor)):
                for j in range(len(Label_Tensor[i])):
                    if Label_Tensor[i, j] == 0:
                        Label_Tensor[i, j] = -100

            Final_Output = self.Final_output_layer(inputs_embeds=Encoder_Decoder_Self_Attention_Layer_output, labels=Label_Tensor)

            return Final_Output



