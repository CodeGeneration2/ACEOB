# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

First, you need to extract the ECG.rar archive file to the current folder.

We give a brief description of the 4 models below.


## E_code
The E_code model makes our main model and its code is in the E_code folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py  \
    --save-dir=/path/to/save_dir  \
    --load=/path/to/model \  # Can be used to restart from checkpoint
    --apps-train-files ~/apps/train \
    --apps-dataroot ~/apps/train/ \
    --grad-acc-steps=8 \
    --epochs=10 \
    --fp16 \
    --deepspeed deepspeed_config.json \
    --batch-size-per-replica=2





