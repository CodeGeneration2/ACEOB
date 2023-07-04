# coding = UTF-8
import os

import torch
from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from Dataset_prompt_NPI import MyDatasetFunction

import wandb

wandb.login(key="")

model_path = 'codet5-base'

# ======================================================== Import Training Data =============================================#
training_data = MyDatasetFunction(
    dataset_path="../../NPI/train",
    max_Token_num=512,
    model_path=model_path
)
test_data = MyDatasetFunction(
    dataset_path="../../NPI/test",
    max_Token_num=512,
    model_path=model_path
)

# ------------------------------------ Tokenization Vocabulary ------------------------------------#
Tokenization_vocabulary = RobertaTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from transformers import Trainer, TrainingArguments

# ############################################################## Training ##################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./trained_model_parameters_{model_path}_NPI-No regression',  # Output Directory
    save_strategy="epoch",  # Model Saving Strategy
    evaluation_strategy="no",  # Evaluation Strategy
    num_train_epochs=10,  # Total epochs
    per_device_train_batch_size=16,  # Training batch size
    gradient_accumulation_steps=2,  # Gradient accumulation
    per_device_eval_batch_size=32,  # Evaluation batch size
    eval_accumulation_steps=1,  # Number of evaluation steps to accumulate output tensors before moving to CPU

    ############################################ predict_with_generate=True,
    save_total_limit=10,  # At most save 2 models
    overwrite_output_dir=True,  # Overwrite
    dataloader_drop_last=True,  # Drop the last
    dataloader_pin_memory=False,  # Whether you want to pin memory in the data loader. Will default to True
    dataloader_num_workers=0,  # Data Loading
    prediction_loss_only=True,  # Only evaluate loss

    fp16=True,

)

trainer = Seq2SeqTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=training_data,  # Training dataset
    eval_dataset=test_data,  # Test dataset
    # data_collator=data_collator,
    tokenizer=Tokenization_vocabulary,

)

trainer.train()
print(
    '\033[0:34m===========================================Training Finished!!!===================================\033[m')
