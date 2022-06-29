# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

First, you need to extract the ECG.rar archive file to the current folder.

We give a brief description of the 4 models below.


## E_code
The E_code model makes our main model and its code is in the E_code folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py


## No expert group E-code 350M
The no expert group E-code 350M model makes our main control model and its code is in the no expert group E-code 350M folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py

## GPT_NEO125M
The GPT_NEO125M model makes our control model and its code is in the GPT_NEO125M folder.

You can run the train.py file directly for training and prediction

We use the following command to run and train.

    train.py

## Run time predictor
The run time predictor model allows us to train the run time predictor and its code is in the Run time predictor folder.

You can run the train.py file directly for training.

We use the following command to run and train.

    train.py
