# coding = UTF-8
import os

import torch
from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from Dataset_prompt_T5 import MyDatasetFunction
from CodeBleu import _bleu
from sacrebleu.metrics import BLEU
import wandb
wandb.login(key="")

model_path = 'codet5-small'

# ======================================================== Load Training Data =============================================#
train_data = MyDatasetFunction(
    dataset_path="../../43-ACEO-temp/train",
    max_token_number=512,
    model_path=model_path
)
test_data = MyDatasetFunction(
    dataset_path="../../43-ACEO-temp/test",
    max_token_number=512,
    model_path=model_path
)


# ------------------------------------ Tokenizer Vocabulary ------------------------------------#
tokenizer_vocab = RobertaTokenizer.from_pretrained(model_path)
# ---------------------------------------- Model ---------------------------------------#
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# ======================================================= Evaluation Metrics Function ==========================================#
def compute_metrics(eval_preds):
    predicted_prob_total_list, label_code_id_total_list = eval_preds

    # ----------------------------------Dimension of predicted_prob_total_list: 【2, sample number, sample length】-----------#
    # ----------------------------------where, 0 is predicted probability, 1 is ground truth-----------#
    predicted_prob_total_list = predicted_prob_total_list[0]

    # --------------------------------------Automatically filled by the system: -100, -100 is not in the vocabulary------------------#
    predicted_prob_total_list[predicted_prob_total_list == -100] = 0
    label_code_id_total_list[label_code_id_total_list == -100] = 0

    print(f'\033[0:34m----------------Length of predicted probabilities: {len(predicted_prob_total_list)} equals {len(label_code_id_total_list)}: length of label code ids\033[m')



    # -------------------------------batch_decode: enter nested list【【1,3,】，【】】， output flat list【def】--------------#
    predicted_code_total_list = tokenizer_vocab.batch_decode(predicted_prob_total_list, skip_special_tokens=True)
    label_code_total_list = tokenizer_vocab.batch_decode(label_code_id_total_list, skip_special_tokens=True)



    # ---------------------------Initialize score lists----------------------#
    bleu_score_list=[]
    Codebleu_score_list=[]
    for i in range(len(label_code_total_list)):
        predicted_code = predicted_code_total_list[i]
        label_code = label_code_total_list[i]



        bleu = BLEU()
        bleu_score = bleu.corpus_score([predicted_code], [[label_code]]).score
        Codebleu_score = round(_bleu(label_code, predicted_code), 2)


        bleu_score_list.append(bleu_score)
        Codebleu_score_list.append(Codebleu_score)

    return {
        'bleu_score': mean(bleu_score_list),
        'Codebleu_score': mean(Codebleu_score_list)
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


from transformers import Trainer, TrainingArguments
# ############################################################## Training ##################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./trained_model_parameters_{model_path}_prompt',          # Output directory
    save_strategy="epoch",                 # Model saving method
    evaluation_strategy="no",            # Prediction method
    num_train_epochs=10,                   # Total epochs
    per_device_train_batch_size=32,        # Training batch size
    # gradient_accumulation_steps=32,       # Gradient accumulation
    per_device_eval_batch_size=32,         # Prediction batch size
    eval_accumulation_steps=1,            # Number of steps for output tensor accumulation before transferring to CPU
    warmup_steps=500,                     # Number of warm up steps
    weight_decay=0.01,                    # Strength of weight decay
    logging_dir='./logs',                  # Log directory
    logging_strategy="epoch",             # Log saving strategy
    logging_first_step=True,              # Log saving first step
    # eval_steps=0,
    ############################################ predict_with_generate=True,
    save_total_limit=1,                     # Save at most 2 models
    overwrite_output_dir=True,              # Overwrite
    dataloader_drop_last=True,              # Discard the last one
    dataloader_pin_memory=False,            # Whether to pin memory in data loader. It will default to True
    dataloader_num_workers=0,               # Data loading
    prediction_loss_only=True,              # Evaluate loss only

    fp16=True,


)


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_vocab, mlm=False)


trainer = Seq2SeqTrainer(
    model=model,                           # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments, defined above
    train_dataset=train_data,                # Training dataset
    eval_dataset=test_data,                 # Test dataset
    # data_collator=data_collator,
    tokenizer=tokenizer_vocab,
    compute_metrics=compute_metrics,         # Evaluation metrics function
    preprocess_logits_for_metrics=preprocess_logits_for_metrics         # Function to preprocess logits before caching them at each evaluation step
)


trainer.train()
print('\033[0:34m===========================================Training Ended!!!===================================\033[m')


trainer.save_model(f'backup_model_parameters_{model_path}_prompt')
