# coding = UTF-8
import os

import torch
from numpy import mean
from transformers import AutoTokenizer, AutoModelForCausalLM

from Dataset_prompt_gpt import MyDatasetFunction
from CodeBleu import _bleu
from sacrebleu.metrics import BLEU
import wandb
wandb.login(key="")


model_path = r'PolyCoder-0.4B'
model_name = 'PolyCoder-0.4B'

# ======================================================== Load Training Data =============================================#
train_data = MyDatasetFunction(
    dataset_path=r"./43-ACEO/train",
    max_token_number=510,
    model_path=model_path
)
test_data = MyDatasetFunction(
    dataset_path=r"./43-ACEO/test",
    max_token_number=510,
    model_path=model_path
)

# ------------------------------------ Vocabulary ------------------------------------#
if "codegen" in model_path:
    # ------------------------------------ Tokenizer Vocabulary ------------------------------------#
    tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)
    tokenizer_vocab.pad_token = tokenizer_vocab.eos_token
elif "incoder" in model_path:
    # ------------------------------------ Tokenizer Vocabulary ------------------------------------#
    tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)
    tokenizer_vocab.pad_token = "<pad>"
elif "PolyCoder-0.4B" in model_path:
    # ------------------------------------ Tokenizer Vocabulary ------------------------------------#
    tokenizer_vocab = AutoTokenizer.from_pretrained(model_path)
    tokenizer_vocab.pad_token = tokenizer_vocab.eos_token


# ---------------------------------------- Model ---------------------------------------#
model = AutoModelForCausalLM.from_pretrained(model_path)
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



    # -------------------------------batch_decode: enter nested list【【1,3,】,【】】, output flat list【def】--------------#
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
        'Codebleu_score': mean(Codebleu_score)
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
training_args = TrainingArguments(
    output_dir=f'./trained_model_parameters_{model_name}_prompt',          # output directory
    save_strategy="epoch",                 # model saving method
    evaluation_strategy="no",            # prediction method
    num_train_epochs=10,                   # total epochs
    per_device_train_batch_size=8,        # train batch size
    gradient_accumulation_steps=4,       # gradient accumulation
    per_device_eval_batch_size=32,         # evaluation batch size
    eval_accumulation_steps=1,            # number of steps for prediction output tensor accumulation before transferring to CPU
    warmup_steps=500,                     # number of warm up steps
    weight_decay=0.01,                    # strength of weight decay
    logging_dir='./logs',                  # log directory
    logging_strategy="epoch",             # log saving strategy
    logging_first_step=True,              # log the first step
    save_total_limit=1,                     # save at most 2 models
    overwrite_output_dir=True,              # overwrite
    dataloader_drop_last=True,              # discard the last one
    dataloader_pin_memory=False,            # pin memory in data loader. It will default to True
    dataloader_num_workers=0,               # data loading
    prediction_loss_only=True,              # evaluate loss only

    fp16=True,


)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_vocab, mlm=False)


trainer = Trainer(
    model=model,                           # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_data,                # training dataset
    eval_dataset=test_data,                 # test dataset
    data_collator=data_collator,
    tokenizer=tokenizer_vocab,
    compute_metrics=compute_metrics,         # evaluation metrics function
    preprocess_logits_for_metrics=preprocess_logits_for_metrics         # function to preprocess logits before caching them at each evaluation step
)


trainer.train()
print('\033[0:34m===========================================Training ended!!!===================================\033[m')
print('\033[0:34m===========================================Training ended!!!===================================\033[m')
print('\033[0:34m===========================================Training ended!!!===================================\033[m')

trainer.save_model(f'backup_model_parameters_{model_name}_prompt')
