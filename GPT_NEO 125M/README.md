# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

First, you need to extract the ECG.rar archive file to the current folder.

We provide a brief description of this GPT_NEO 125M model below.



## GPT_NEO 125M
The GPT_NEO 125M model makes our control model and its code is in the GPT_NEO125M folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py  \
    --epochs=30  \
    --batch-size=1  \
    --gradient_accumulation=32  \
    --Generated_models=Generated_models



