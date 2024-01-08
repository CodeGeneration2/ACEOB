# coding = UTF-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from Dataset import MyDatasetFunction  # Translated tasetFunction
from transformers import DataCollatorForLanguageModeling


model_path = 'codet5-base'  # Translated  to model_path

# ======================================================== Import Training Data =============================================#
training_data = MyDatasetFunction(  # Translated nd its parameters
    dataset_path="../../runtime_training_set/train",  # Translatedto dataset_path
    max_token_count=512,  # Translatedto max_token_count
    model_path=model_path,  # Translated to model_path
    train_or_predict="train"  # Translated  to train_or_predict
)

# ------------------------------------ Tokenizer Vocabulary ------------------------------------#
tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)  # Translated to tokenizer_vocab
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)  # Translated o model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments
# ############################################################## Training ##################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./trained_models_{model_path}_prompt_runtime_predictor',  # Translatedned_models and output_dir respectively
    save_strategy="epoch",  # Save model method
    evaluation_strategy="no",  # Prediction method
    num_train_epochs=10,  # Total epochs
    per_device_train_batch_size=8,  # Training batch size
    gradient_accumulation_steps=4,  # Gradient accumulation
    per_device_eval_batch_size=32,  # Prediction batch size
    eval_accumulation_steps=1,  # Number of prediction steps to accumulate before moving the results to the CPU

    warmup_steps=500,  # Warm up step count
    weight_decay=0.01,  # Strength of weight decay
    dataloader_num_workers=0,  # Data loader

    logging_dir='./logs',  # Logging directory
    logging_strategy="epoch",  # Logging save strategy
    logging_first_step=True,  # Save first step logs
    save_total_limit=11,  # Save up to 2 models
    overwrite_output_dir=True,  # Overwrite
    dataloader_drop_last=True,  # Discard the last
    dataloader_pin_memory=False,  # If you want to pin memory in data loader. Will default to True

    prediction_loss_only=True,  # Only evaluate loss

    fp16=True,
)

# =============================================================================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer_vocab,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=model,  # The instantiated 🤗 Transformers model to be trained
    args=training_args,  # Training arguments, defined above
    train_dataset=training_data,  # Training set
    data_collator=data_collator,
    tokenizer=tokenizer_vocab,
)

trainer.train()
print('\033[0:34m=========================================== Training Finished !!! ===================================\033[m')
