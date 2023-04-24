import os
import numpy as np
import torch
import transformers
from numpy import mean
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from Dataset_prompt import Dataset
from CodeBleu import _bleu
from sacrebleu.metrics import BLEU


Training_Data = Dataset(Path_to_dataset='../../ECG/train', Maximum_number_of_tokens=2048, Model_path='codet5-base')
Test_data = Dataset(Path_to_dataset='../../ECG/test', Maximum_number_of_tokens=2048, Model_path='codet5-base')
Tokenized_word_list = RobertaTokenizer.from_pretrained('codet5-base')
model = T5ForConditionalGeneration.from_pretrained('codet5-base')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model.to(device)

def compute_metrics(eval_preds):
    (Total_list_of_predicted_probabilities, Total_list_of_tag_code_ids) = eval_preds
    Total_list_of_predicted_probabilities = Total_list_of_predicted_probabilities[0]
    Total_list_of_predicted_probabilities[(Total_list_of_predicted_probabilities == (- 100))] = 0
    Total_list_of_tag_code_ids[(Total_list_of_tag_code_ids == (- 100))] = 0
    print(f'[0:34m-------------{len(Total_list_of_predicted_probabilities)} == {len(Total_list_of_tag_code_ids)}')
    Total_list_of_prediction_codes = Tokenized_word_list.batch_decode(Total_list_of_predicted_probabilities, skip_special_tokens=True)
    Total_list_of_tag_codes = Tokenized_word_list.batch_decode(Total_list_of_tag_code_ids, skip_special_tokens=True)
    List_of_bleu_scores = []
    Codebleu_score_list = []
    for i in range(len(Total_list_of_tag_codes)):
        Prediction_code = Total_list_of_prediction_codes[i]
        Tag_Codes = Total_list_of_tag_codes[i]
        bleu = BLEU()
        bleu_score = bleu.corpus_score([Prediction_code], [[Tag_Codes]]).score
        Codebleu_score = round(_bleu(Tag_Codes, Prediction_code), 2)
        List_of_bleu_scores.append(bleu_score)
        Codebleu_score_list.append(Codebleu_score)
    return {'bleu': mean(List_of_bleu_scores), 'Codebleu': mean(Codebleu_score)}

def preprocess_logits_for_metrics(logits, labels):
    '\n    Original Trainer may have a memory leak.\n    This is a workaround to avoid storing too many tensors that are not needed.\n    '
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=(- 1))
    return (pred_ids, labels)
data_collator = DataCollatorForSeq2Seq(Tokenized_word_list, model=model)
training_args = Seq2SeqTrainingArguments(output_dir='./codeT5_base', save_strategy='epoch', evaluation_strategy='no', num_train_epochs=10, per_device_train_batch_size=1, gradient_accumulation_steps=32, per_device_eval_batch_size=1, eval_accumulation_steps=32, warmup_steps=500, weight_decay=0.01, logging_dir='./log', logging_strategy='epoch', logging_first_step=True, save_total_limit=1, overwrite_output_dir=True, dataloader_drop_last=True, dataloader_pin_memory=False, dataloader_num_workers=0, fp16=True)
trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=Training_Data, eval_dataset=Test_data, data_collator=data_collator, tokenizer=Tokenized_word_list, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
trainer.train()

trainer.evaluate()

