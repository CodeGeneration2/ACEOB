import torch
import os
from numpy import mean
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from Dataset_prompt import Dataset
from CodeBleu import _bleu
from sacrebleu.metrics import BLEU, CHRF, TER


Test_data = Dataset(Path_to_dataset='../ECG/test', Maximum_number_of_tokens=2048, Model_path='codeT5_large/checkpoint-7510')
Tokenized_word_list = AutoTokenizer.from_pretrained('codeT5_large/checkpoint-7510')
model = T5ForConditionalGeneration.from_pretrained('codeT5_large/checkpoint-7510')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model.to(device)
if (not os.path.exists(f'generate')):
    os.makedirs(f'generate')
Total_list_of_BLEU_scores = []
Total_list_of_CodeBLEU_scores = []
for (Index, test_tensor) in tqdm(enumerate(Test_data)):
    Generate_Token = model.generate(torch.unsqueeze(test_tensor['input_ids'].to(device), dim=0), max_length=768, num_beams=5, early_stopping=True)
    Prediction_code = Tokenized_word_list.decode(Generate_Token[0], skip_special_tokens=True)
    Standard_answer = test_tensor['Label_Code']
    bleu = BLEU()
    bleu_score = bleu.corpus_score([Prediction_code], [[Standard_answer]]).score
    bleu_score = round(bleu_score, 2)
    Total_list_of_BLEU_scores.append(bleu_score)
    Codebleu_score = round(_bleu(Standard_answer, Prediction_code), 2)
    Total_list_of_CodeBLEU_scores.append(Codebleu_score)
    with open(f'generate/{Index},CodeBLEU{Codebleu_score},bleu{bleu_score}.txt', 'w', encoding='UTF-8') as f:
        f.write(Prediction_code)
    with open(f'generate/{Index},ans.txt', 'w', encoding='UTF-8') as f:
        f.write(Standard_answer)
Average_BLEU_Score = mean(Total_list_of_BLEU_scores)
Average_CodeBLEU_Score = mean(Total_list_of_CodeBLEU_scores)
with open(f'BLEU,BLEU,{Average_BLEU_Score:.3f},{len(Total_list_of_BLEU_scores)}.txt', 'w', encoding='UTF-8') as f:
    f.write(str(Total_list_of_BLEU_scores))
with open(f'CodeBLEU,{Average_CodeBLEU_Score:.3f},{len(Total_list_of_CodeBLEU_scores)}.txt', 'w', encoding='UTF-8') as f:
    f.write(str(Total_list_of_CodeBLEU_scores))
print(f'[0:34m========== {len(Total_list_of_BLEU_scores)} == {len(Total_list_of_CodeBLEU_scores)} ==[m')
print(f'[0:34m=========={Average_CodeBLEU_Score:.3f}      {Average_BLEU_Score:.3f} ==[m')