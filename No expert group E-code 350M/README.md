# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

First, you need to extract the ECG.rar archive file to the current folder.

We provide a brief description of this No expert group E-code 350M model below.

    
    
## No expert group E-code 350M
The no expert group E-code 350M model makes our main control model and its code is in the no expert group E-code 350M folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py
    --epochs=30  \
    --batch-size=1  \
    --gradient_accumulation=32  \
    --Generated_models=Generated_models

