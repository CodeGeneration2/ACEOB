# -*- coding: utf-8 -*-

import torch
import os
# from numpy import mean
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
# from CodeBleu score import _bleu
# from sacrebleu.metrics import BLEU, CHRF, TER
import wandb
wandb.login(key="82cfc67dbc70c15d489f7eaaae135311f3716c1a")
print('\033[0:34m===========================================Go for it!========================================\033[m')

from Dataset_prompt_gpt import MyDatasetFunction
model_path = "TrainedModelParams_PolyCoder-0.4B_prompt/checkpoint-26900"
model_name = "PolyCoder-0.4B(GPT)"
beam_sample_size = 10
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ======================================================== Import training data =============================================#
test_data = MyDatasetFunction(
    dataset_path="./43-ACEO/test",
    max_token_count=510,
    model_path=model_path,
    include_path=True,
)

# ------------------------------------ Tokenizer vocabulary ------------------------------------#
tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)

# ------------------------------------ Vocabulary ------------------------------------#
if "codet5" in model_path:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
elif "codegen" in model_path:
    tokenizer_vocab.pad_token = tokenizer_vocab.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
elif "PolyCoder" in model_path:
    tokenizer_vocab.pad_token = tokenizer_vocab.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)

# ---------------------------------------- Model ---------------------------------------#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# ======================================= Create model-generated code output directory ======================#
if not os.path.exists(f"{model_name}_generate_beams_{beam_sample_size}"):
    os.makedirs(f"{model_name}_generate_beams_{beam_sample_size}")


# ---------------------------------- Compute Bleu ----------------------------------------#
BLEU_score_total_list = []
CodeBLEU_score_total_list = []


for index, test_tensor in tqdm(enumerate(test_data)):
    try:
        # Mistake:!!! GPT output includes input part !!!
        if beam_sample_size == "use_sampling_method":
            generated_token_list = model.generate(test_tensor['input_ids'].to(device),
                                  attention_mask=test_tensor['attention_mask'].to(device),
                                  max_new_length=510,          # Mistake:max_length refers to the total length of input+output
                                  min_new_length=4,
                                  num_return_sequences=9,  # Set the number of return sequences
                                  no_repeat_ngram_size=2,               # Optional parameter to avoid repeating n-gram
                                  early_stopping=True,                 # Optional parameter, stop searching early to improve efficiency
                                  temperature=0.8,
                                  do_sample=True,
                                  top_k=50,  # Set top-k value  or 30
                                  top_p=0.95,  # Set top-p value   or 0.9
                                  )
        else:
            # =================================================== simply generate a single sequence =
            with torch.no_grad():
                generated_token_list = model.generate(test_tensor['input_ids'].to(device),
                                      attention_mask=test_tensor['attention_mask'].to(device),
                                      max_new_length=510,                  # Mistake:max_length refers to the total length of input+output
                                      min_new_length=8,
                                      num_beams=beam_sample_size,                 # Set beam width
                                      num_return_sequences=beam_sample_size,      # Set the number of return sequences
                                      no_repeat_ngram_size=2,               # Optional parameter to avoid repeating n-gram
                                      early_stopping=True,                 # Optional parameter, stop searching early to improve efficiency
                                      )

        # Code path: f"{self.dataset_path}/{question}/{content}"
        code_path = test_tensor["code_path"]
        question_index = code_path.split("/test/")[-1].split("/high_efficiency-low_efficiency_code_pair")[0]
        high_low_efficiency_code_pair_index = code_path.split("/high_efficiency-low_efficiency_code_pair")[-1]

        generated_code_storage_path = f"./{model_name}_generate_beams_{beam_sample_size}/{question_index}/high_efficiency-low_efficiency_code_pair{high_low_efficiency_code_pair_index}"
        if not os.path.exists(generated_code_storage_path):
            os.makedirs(generated_code_storage_path)

        # ======================================= Mistake:!!! GPT output includes input part !!! =========================#
        input_length = len(test_tensor['input_ids'][0])
        for code_number, generated_token in enumerate(generated_token_list):
            generated_token = generated_token[input_length:]
            predicted_code = tokenizer_vocab.decode(generated_token, skip_special_tokens=True)

            # ---------------------------- Save record file -----------------------------------------------------------#
            with open(f"./{generated_code_storage_path}/{code_number}.txt", 'w', encoding='UTF-8') as f:
                f.write(predicted_code)
    except:
        print(f"!!!Error!!!{test_tensor['code_path']}")
        continue
