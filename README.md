# Efficient-Code-Generation-with-GEC


## How to Use

### Implementation Train the model -> predict the generated code -> perform test on the generated code


## The GEC Dataset(https://github.com/CodeGeneration2/GEC-Dataset)
  
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

Although our initial intent in creating the GEC dataset was to evaluate the ability to select efficient algorithms, the GEC dataset is actually a versatile dataset that can be applied to various tasks. For instance, the efficient-inefficient code pairs and the efficient code from multiple algorithm schemes could be of value to research in code generation and code cloning. Consequently, we derived two datasets from the GEC benchmark, namely GEC-CG(https://github.com/CodeGeneration2/GEC-CG-DataSet) and GEC-clone(https://github.com/CodeGeneration2/GEC-clone-DataSet), with the aim of promoting more innovative research in the fields of code generation and code cloning.

The GEC-CG dataset contains only natural language (NL) descriptions and their corresponding valid ground-truth code, featuring a structure similar to the APPS dataset.

Each data point in the GEC-clone dataset includes two distinct code implementations. Although this paper does not focus on code cloning research, the GEC dataset provides different code implementations for the same functionality for each problem. Notably, the GEC-clone dataset is specifically designed for semantic clone research. In other words, the GEC-clone dataset aims to investigate functional similarities between two code implementations.

## The predictor

To calculate the RES scores for partial codes, we trained a code running time predictor using CodeT5-base to predict the running time of the code. The fine-tuning settings for this predictor were the same as those for the previously mentioned CodeT5 models. The predictor is fine-tuned using the GEC dataset. Unlike the GEC task, the “input feature” of the predictor refers to code, and the “label” used for gradient propagation is the code execution time (obtained from the codeforces website).

The model parameters for the runtime predictor are here https://drive.google.com/file/d/1YRXfFuzHq1TsNMpAivPYaUySsbSV2Nmt/view?usp=sharing.
