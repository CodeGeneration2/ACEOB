# Training prediction code
Since our experiments have one main model, two control models and a runtime predictor. Therefore, we placed the four model codes separately.

We give a brief description of the 4 models below.


## E_code
The E_code model makes our main model and its code is in the E_code folder.

We use the following command to run and train.

    train.py


## No expert group E-code 350M
The no expert group E-code 350M model makes our main control model and its code is in the no expert group E-code 350M folder.

We use the following command to run and train.

    train.py

## GPT_NEO125M
The GPT_NEO125M model makes our control model and its code is in the GPT_NEO125M folder.

We use the following command to run and train.

    train.py

## Run time predictor
The run time predictor model allows us to train the run time predictor and its code is in the Run time predictor folder.

We use the following command to run and train.

    train.py
