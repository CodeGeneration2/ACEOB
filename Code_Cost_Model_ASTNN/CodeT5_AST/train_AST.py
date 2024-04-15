# coding = UTF-8
import os

import torch

from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from Dataset_NPI_AST import MyDatasetFunction
from transformers import DataCollatorForLanguageModeling


model_path = 'codet5-base'


# ======================================================== Import Training Data =============================================#
training_data = MyDatasetFunction(
    dataset_path="../ACEOB-NPI/train",
    max_token_count=512,
    model_path=model_path,
    train_or_predict="train"
)

# ------------------------------------ Tokenization Vocabulary ------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


from transformers import Trainer, TrainingArguments
# ############################################################## Training ##################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./{model_path}_predict_time',        # Output directory
    save_strategy="epoch",                           # Save model strategy
    evaluation_strategy="no",                        # Prediction method
    num_train_epochs=10,                             # Total epochs
    per_device_train_batch_size=32,                  # Training batch size
    gradient_accumulation_steps=1,                   # Gradient accumulation
    per_device_eval_batch_size=32,                   # Prediction batch size
    eval_accumulation_steps=1,                       # Number of prediction steps to accumulate before moving the results to the CPU

    warmup_steps=500,                                # Warm up steps
    weight_decay=0.01,                               # Strength of weight decay
    dataloader_num_workers=0,                        # Data loader workers

    logging_dir='./logs',                            # Logging directory
    logging_strategy="epoch",                        # Logging saving strategy
    logging_first_step=True,                         # Log the first step
    save_total_limit=11,                             # Maximum number of models to save
    overwrite_output_dir=True,                       # Overwrite output directory
    dataloader_drop_last=True,                       # Drop the last incomplete batch
    dataloader_pin_memory=False,                     # Whether to pin memory in data loader. Will default to True

    prediction_loss_only=True,                       # Only evaluate the loss

    fp16=True,                                       # Use fp16

)

# =============================================================================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=model,                        # The instantiated 🤗 Transformers model to be trained
    args=training_args,                 # Training arguments, defined above
    train_dataset=training_data,        # Training dataset

    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
