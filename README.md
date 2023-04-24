# Efficient-Code-Generation-with-GEC


## How to Use

### Implementation Train the model -> predict the generated code -> perform test on the generated code
#### To use the GEC source code extremely fast: 

1. Extract the ECG dataset to the E_code folder and change the file name to ECG. 
2. Run the train.py file. 

#### Fast-running classification experiments: 

Set Command_line_parameters.task = 0 to train the E-code model.

Set Command_line_parameters.task = 0 and set Command_line_parameters.RELU = 1 to train a comparison experiment using the RELU activation function.

Set Command_line_parameters.task = 0 and set Command_line_parameters. heads = 8 to train a comparison experiment using 8 heads.

Set Command_line_parameters.task = 1 to train the No-expert-E-code model.

Set Command_line_parameters.task = 2 to train the GPT model.

#### Extremely fast use of Time_Predictor source code: 
1. Extract the ECG dataset to the E_code folder and change the file name to ECG. 
2. Run the train.py file to train the model.

3. Put the code to be predicted into Code_to_be_predicted a
4. Run Prediction_generation_code to automatically predict the code runtime.


## The GEC Dataset(https://github.com/CodeGeneration2/ECG-dataset)
  
The GEC (Generation of Efficient Code) dataset is composed of problems from the open programming website CodeforRESs, as the GEC benchmark aims to evaluate how human programmers can improve their code efficiency. Human programmers can understand programming problems and approaches through natural language descriptions and inefficient code. Finally, the model's ability to generate efficient code is assessed using NMMCB scores, RES scores, and IO unit tests.

The Generation of Efficient Code benchmark, abbreviated as GEC, includes a total of:

3,712 coding problems,
31,577 efficient-inefficient code pairs for fine-tuning the model,
13,092 efficient codes utilizing different solution algorithms for calculating NMMCB scores.

The GEC dataset offers an accurate and comprehensive approach to efficient code generation. Each efficient-inefficient code pair helps the model learn an efficiency optimization method. Furthermore, the efficient-inefficient code pairs in the GEC dataset also include efficient codes with different algorithmic solutions, providing a crucial reference for the NMMCB metric (Section 5.2). As a result, as long as the model-generated code is sufficiently efficient, it can achieve a good score, even if it is entirely different from the ground truth code. The problems in the GEC dataset are challenging and complex, with the length of the problem typically being proportional to the difficulty. On average, the problems have a length of approximately 351 tokens. If a model performs well on the GEC dataset, it suggests that the model has mastered various algorithms and code optimization strategies and possesses the ability to flexibly apply data structures and programming techniques.

The Codeforces platform categorizes problems into 28 difficulty levels. Clearly, this is overly complex. Additionally, some high-difficulty problems lack the required solution code. Therefore, we classified these 28 difficulty levels into three categories based on the size of the data volume and the proportion of difficulty levels in the GEC dataset. They are "easy" (difficulty 0-3), "medium" (difficulty 4-11), and "hard" (difficulty 12-27).

Easy: These problems can be solved by most programmers with 1-2 years of experience and do not require complex algorithms. The average efficient code in the test set consists of only 13 lines, and efficiency may be optimized with just a single loop reduction. There are 9,445 easy-level efficient-inefficient code pairs, with 951 pairs designated as the test set.

Medium: These problems may involve more algorithms and more direct issues. Examples of such problems include data structures, such as trees or graphs, or questions requiring non-trivial algorithms. The test set has an average of 26 lines of efficient code, requiring proficient mastery of various algorithm principles to optimize efficiency. There are 18,348 medium-level efficient-inefficient code pairs, with 1,792 pairs designated as the test set.

Hard: These problems are the most challenging and reach the level of state-of-the-art programming competitions. The test set has an average of 38 lines of efficient code, requiring proficient mastery of code efficiency for all algorithms to optimize effectively. There are 3,784 hard-level efficient-inefficient code pairs, with 342 pairs designated as the test set.

Although our initial intent in creating the GEC dataset was to evaluate the ability to select efficient algorithms, the GEC dataset is actually a versatile dataset that can be applied to various tasks. For instance, the efficient-inefficient code pairs and the efficient code from multiple algorithm schemes could be of value to research in code generation and code cloning. Consequently, we derived two datasets from the GEC benchmark, namely GEC-CG(https://github.com/CodeGeneration2/ECG-dataset) and GEC-clone(https://github.com/CodeGeneration2/ECG-dataset), with the aim of promoting more innovative research in the fields of code generation and code cloning.

The GEC-CG dataset contains only natural language (NL) descriptions and their corresponding valid ground-truth code, featuring a structure similar to the APPS dataset.

Each data point in the GEC-clone dataset includes two distinct code implementations. Although this paper does not focus on code cloning research, the GEC dataset provides different code implementations for the same functionality for each problem. Notably, the GEC-clone dataset is specifically designed for semantic clone research. In other words, the GEC-clone dataset aims to investigate functional similarities between two code implementations.




## Diagrammatic figure
In the Efficient-Code-Generation-with-E-Code work, the diagrammatic figure is in the [Diagrammatic figure folder](https://github.com/CodeGeneration2/Diagrammatic-figure/tree/main/Diagrammatic%20figure).



## Generated code has been predicted
In the Efficient-Code-Generation-with-E-Code work, the authors use a fine-tuned pre-trained model to predict a range of codes to be generated. 
Due to the need for comparative experiments, three code generation models are available. 
The three code generation models are E-code 350M, GPT-Neo 125M, and No expert group E-code 350M. 
We use each of the three fine-tuned code generation models to generate codes. 
Below we have [the code generated by the three fine-tuned code generation models](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted).


### E-code 350M
We give [the results of 3 times code generation in the E-code 350M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/E-code%20350M).


### GPT-Neo 125M
We give [the case results of one code generation for the GPT-Neo 125M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/GPT-Neo%20125M).


### No expert group E-code 350M
We give [the case results of one code generation for the no expert group E-code 350M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/No%20expert%20group%20E-code%20350M).

