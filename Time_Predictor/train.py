import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import transformers
from numpy import mean
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from Dataset import Dataset

import numpy as np


Training_Data = Dataset(Data_set_paths='../../ECG', Maximum_Number_of_Tokens=2048, Model_path='codet5-base')
Tokenized_word_list = RobertaTokenizer.from_pretrained('codet5-base')
model = T5ForConditionalGeneration.from_pretrained('codet5-base')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model.to(device)

def preprocess_logits_for_metrics(logits, labels):
    '\n    Original Trainer may have a memory leak.\n    This is a workaround to avoid storing too many tensors that are not needed.\n    '
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=(- 1))
    return (pred_ids, labels)
data_collator = DataCollatorForSeq2Seq(Tokenized_word_list, model=model)
training_args = Seq2SeqTrainingArguments(output_dir='./codeT5_base', save_strategy='epoch', evaluation_strategy='no', num_train_epochs=10, per_device_train_batch_size=1, gradient_accumulation_steps=32, per_device_eval_batch_size=1, eval_accumulation_steps=32, warmup_steps=500, weight_decay=0.01, logging_dir='./log', logging_strategy='epoch', logging_first_step=True, save_total_limit=1, overwrite_output_dir=True, dataloader_drop_last=True, dataloader_pin_memory=False, dataloader_num_workers=0, prediction_loss_only=True, fp16=True)
trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=Training_Data, data_collator=data_collator, tokenizer=Tokenized_word_list, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
trainer.train()
