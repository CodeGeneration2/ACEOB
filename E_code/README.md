# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

First, you need to extract the ECG.rar archive file to the current folder.

We give a brief description of the 4 models below.


## E_code
The E_code model makes our main model and its code is in the E_code folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py  \
    --device=0  \
    --epochs=30  \
    --batch-size=1  \
    --gradient_accumulation=32  \
    --Generated_models=Generated_models






