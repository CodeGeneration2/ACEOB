# Measuring Code Efficiency Optimization Capabilities with ACEOB

![论文流程图-2](https://github.com/CodeGeneration2/ACEOB/assets/95161813/0fed48ab-d004-4379-a79a-3ed70244e975)


## A B S T R A C T

As Moore’s Law gains diminish, software performance and efficiency become increasingly vital.
Optimizing code efficiency is challenging, even for professional programmers. However, related
research remains relatively scarce, and rigorously assessing models’ abilities to optimize code
efficiency is fraught with difficulties. More concerning, recent large language models often
exhibit “slow positives” in code generation, failing to meet the real-time efficiency requirements
of algorithms. In response to this challenge, we first conduct an in-depth analysis of “code
patterns” in the model training dataset, meticulously exploring human-written code. Secondly,
we define a task for optimizing code efficiency and introduce the Automatic Code Efficiency
Optimization Benchmark (ACEOB), which consists of 95,359 pairs of efficient-inefficient code
aimed at assessing code efficiency optimization capabilities. To our knowledge, ACEOB is the
first dataset specifically targeting Python code efficiency optimization. To evaluate models’
ability in optimizing code efficiency, we propose two new metrics: the Isomorphic Optimal
Comparison CodeBLEU (IOCCB) metric and the Normalized Performance Index (NPI) metric,
to assess the efficiency of model-generated code. We also evaluate several advanced code
generation models, such as PolyCoder and CodeT5, after fine-tuning them on ACEOB and
demonstrate that the efficiency of each model improves after introducing the NPI filter. However,
we note that even the most advanced AI models currently, including ChatGPT, are still not
fully satisfactory in code efficiency optimization tasks. Our dataset and models are available
at: https://github.com/CodeGeneration2/ACEOB.


## Highlights

• Dataset: We developed the ACEOB benchmark dataset, currently the first targeted at competition-level Python code efficiency optimization. We have also made public the ACEOB-Ori and ACEOB-NPI datasets.

• Motivation: We analyzed the “code patterns” AI learns and the reasons behind generating inefficient code.

• Metrics \& Cost Models: We introduced the IOCCB and NPI evaluation metrics and developed cost models for predicting Python code execution time and Python code NPI scores.

• Experimentation: We tested the performance of mainstream code models on the IC2EC task and demonstrated the significant effect of the NPI filter in enhancing model performance.


## Example of IC2EC task. 

Each row corresponds to an efficient-inefficient code pair, consisting of inefficient code
(long running time) and efficient code (short running time).

![原始绘图数据6 drawio (1)](https://github.com/CodeGeneration2/ACEOB/assets/95161813/1b1aa5a4-e820-4c91-9052-6b28b02a168b)




## IOCCB

![CodeBLEU-MAX-5](https://github.com/CodeGeneration2/ACEOB/assets/95161813/2f479292-5b12-4866-a375-7a462cf78926)

IOCCB Score Calculation Process. This figure shows the detailed calculation process of the IOCCB score. The
process begins with inputting the IC from the Algorithm Father-Son Pair into LLMs to generate the predicted code 𝑔.
Then, by matching the generated code 𝑔 with ECs and alternate efficient codes to calculate the CodeBLEU score, forming
the set 𝑂. Additionally, each code will be standardized in terms of variables and function names before matching, forming
the set 𝑆. Next, we calculate the average of set 𝑂 (𝑂𝑎𝑣𝑔 ), the average of set 𝑆 (𝑆𝑎𝑣𝑔 ), and the maximum of set 𝑆 (𝑆𝑚𝑎𝑥).
Finally, the IOCCB score is defined as the maximum of set 𝑆 plus the square root of the difference between the average
of set 𝑆 and set 𝑂, i.e., 𝐵𝑚𝑎𝑥 + √(𝐵𝑎𝑣𝑔 − 𝐴𝑎𝑣𝑔 ).

## NPI

![fig-NPI-Calculation](https://github.com/CodeGeneration2/ACEOB/assets/95161813/6d9f9ddb-f6e2-4a79-82c3-44c120259131)

NPI Metric Calculation Process. This figure details the calculation process of the NPI metric, which is implemented
in two steps. First, the execution time’s median and minimum values are mapped to the scoring range [50, 100]. Then,
the execution time’s maximum value and median are mapped to the scoring range [0, 50]. The three key points, 0, 50,
and 100, represent the maximum execution time, median execution time, and minimum execution time, respectively.



## Dataset

![数据集流程图-6](https://github.com/CodeGeneration2/ACEOB/assets/95161813/85487a1c-c56b-46fa-ad47-7f318e091a68)

### [**ACEOB-Ori Dataset**](https://drive.google.com/file/d/1ANQB85mwh8lspJ3yx80Y5pEvpuHSV8wv/view?usp=sharing)


We utilized the data collected to systematically assemble the ACEOB-Ori dataset.
The Automatic Code Efficiency Optimization Benchmark Original (ACEOB-Ori) comprises:

• A total of 5,262 problems. Each problem includes a Natural Language description (incorporating the problem
statement, time/space constraints, I/O description, and I/O unit test cases/explanation), the URL of the problem
source, and the URL of the code source. These problems are challenging and complex, given that the average
length of their natural language descriptions is 578 words. Furthermore, we have organized the statistical data
from Section 2 and included it here.

• 901,038 code entries. These code entries were uniformly sampled from all codes on Codeforces, based on their
execution time. Each code entry contains information about its running time, used space, and NPI score.

• I/O unit tests. These are split into public and hidden types. The publicly available I/O unit tests, just like NL,
serve as inputs. The hidden I/O unit tests are utilized to assess the functionality of the code, determining its
capability to accomplish tasks. The public I/O unit tests typically range from 1 to 2, while each problem averages
47 hidden I/O unit tests.

• 36 types of algorithm labels. These algorithm labels represent the recommended algorithmic strategies for
solving the given problems, with examples including math, geometry, and greedy. On average, each problem is
associated with 2.5 algorithm labels.

• 28 levels of difficulty categories. They range from the simplest entry-level difficulty (level 0) to the most
challenging level (level 27).



### [**ACEOB-NPI Dataset**](https://drive.google.com/file/d/1r45TEVEvCsypIJeogZ0mJ3ZMGlb40cV7/view?usp=sharing) : The Training Set for NPI Score Predictor

It is well known that we cannot judge the efficiency of codes implementing different functionalities merely by their
execution times. For instance, a code implementing a simple functionality may have an execution time of 100 ms, while
another code implementing a more complex functionality may take 500 ms. We cannot simply conclude that the former
is more efficient than the latter. Although time complexity can be used to assess algorithmic efficiency, determining
the worst-case time complexity for an algorithm is undecidable. Furthermore, a single piece of code often employs
multiple algorithms, which further complicates matters. In response to the pressing need within the coding efficiency
community for a standardized measure of code efficiency, we propose the NPI score. We plan to train
a model to predict a code’s NPI score, which naturally requires a novel training dataset.
We created the ACEOB-NPI dataset to train models to predict NPI scores. First, we calculated the NPI score for each
code within the dataset, appending it to the code name. We then extracted all the codes from the dataset to compose the
ACEOB-NPI dataset. The Efficiency Decoding and Code Prediction-NPI Score (abbreviated as ACEOB-NPI) dataset
contains a total of 661,496 training entries.


### [ACEOB Dataset](https://drive.google.com/file/d/1eUoOWPPU_2hHeZER5VNc7Xues1uyKhFx/view?usp=sharing)

The ACEOB dataset contains a plethora of efficient-inefficient code pairs. After the clustering filtering in Section
5.7, each problem features a set of inefficient and efficient codes. In the IC2EC task, we believe that the EC should
remain as similar as possible to the IC. Therefore, we combined the inefficient and efficient code sets
into efficient-inefficient code pairs based on CodeBLEU scores, following the concept of “Algorithmic Parent-Child
Pairs” . These code pairs help the model to comprehend code efficiency. However, this pairing method
implies that each IC uniquely corresponds to an EC, a supposition that is flawed. An IC should not have just one code
solution, as different efficient solutions, corresponding to different codes, often exist for a single IC. As a result, we
supplemented each efficient-inefficient code pair with a set of efficient codes using different algorithmic solutions.
These alternative efficient codes were used to calculate the IOCCB scores.

The ACEOB dataset was divided into training and test sets based on time. Randomized splitting methods, such as
those used in APPS, have certain drawbacks because many model’s pre-training datasets already contain their test sets.
Therefore, we split the dataset temporally, designating problems that appeared after May 4, 2022 (the distribution date
of the last model, Polycode) as the test set.

Key components of the Automatic Code Efficiency Optimization Benchmark (ACEOB) dataset include:

• 95522 efficient-inefficient code pairs, 9,415 of which were allocated to the test set.

• I/O unit tests. There are typically 1-2 public I/O unit tests, while hidden I/O unit tests average 39 per data entry.

• Reference efficient codes. Used for calculating IOCCB scores, each data entry averages 40 reference efficient
codes.

• 36 algorithm tags. Algorithm tags recommend the algorithmic approach for solving the problem, e.g., math,
geometry, and greedy. On average, each data entry includes 2.25 algorithm tags. Figure 6 illustrates the
distribution of algorithm labels within the 9,415 samples of the test set from the ACEOB dataset.

• 19 difficulty categories, ranging from the simplest entry-level difficulty 0 to the most challenging difficulty
18 (sourced from the Codeforces website). These difficulties correspond to the complexity of the code’s
functionality. We subdivided them into three higher-level difficulty categories: Introductory (difficulty 0),
Interview (difficulties 1-3), and Competition (difficulties 4-18). In the ACEOB dataset’s test set of 9,415 entries,
these three difficulty levels comprise 5,428 (Introductory), 2,232 (Interview), and 1,755 (Competition) entries
respectively.





## Python Code Cost Model & NPI Filter

Although research on Python code cost models is relatively scarce, they play an important role in the IC2EC task.
When the model optimizes IC, its output must be efficient even if there are functional deficiencies (which may only
require simple manual modifications). Otherwise, even if the code can pass I/O testing, its inefficiency makes it merely
equivalent to another IC, almost without value. Therefore, the value premise of the output results is that they must be
efficient, which requires cost models to assess.


We have trained two major cost models: predicting Python code execution time and [predicting Python code NPI
scores](https://drive.google.com/file/d/1XjWxYjBi6uLs5Pw-EAfzngJvGOaSglCS/view?usp=sharing). Since predicting NPI is more challenging, the cost model accuracy for predicting execution time is higher
than that for predicting NPI. However, the NPI score can reflect more information (such as remaining optimization
space information) and provide effective guidance, which may be more valuable than execution time in some cases.
Therefore, we provide two types of cost models.


The NPI filter is a mechanism that uses NPI scores to filter inefficient codes. In experiments, the NPI filter first
uses the cost model that predicts execution time to calculate the code’s execution time, then calculates the code’s NPI
score using this execution time. Based on the NPI score, the filter will select efficient codes, providing an effective
preliminary filtering method for optimizing inefficient codes.






## Models

In this study, we evaluated the effects of fine-tuning several different LLMs on the ACEOB dataset, including the
CodeGen model, PolyCoder model , and the CodeT5 series models Both the CodeGen and PolyCoder
models are decoder models. The CodeT5 series models are variants based on the T5 model, specifically designed for
code generation. Additionally, we also evaluated the performance of the ChatGPT model on ACEOB. We provide
an overview of the models we used:

• CodeT5-small (60M). As the smallest variant of the CodeT5 models, it offers good performance with low
computational resource demands.

• CodeT5-base (220M). A medium-sized model in the CodeT5 series, it strikes a balance between performance
and computational resource requirements.

• CodeT5-large-ntp-py (770M). This is the largest variant of the CodeT5 models, with additional pre-training
on a Python dataset to focus on generating Python code. Note that this model was proposed by CodeRL.

• CodeGen-mono (350M). The CodeGen model is a code generation model based on GPT-2, with solid
performance and wide-ranging programming language support.

• PolyCoder (0.4B). The PolyCoder model is a novel deep learning model focused on encoding and decoding
multiple programming languages, supporting automated code generation and program understanding.

• ChatGPT. ChatGPT possesses strong capabilities in code generation. It not only understands and interprets
programming requirements but also generates corresponding code snippets according to these requirements, effectively enhancing development efficiency. Particularly in handling common programming tasks and problems,
its prediction and code generation abilities are extremely precise. In this context, we used gpt-3.5-turbo model.



The parameters of the trained model are here.

• [**CodeGen-mono (350M)**](https://drive.google.com/file/d/1hCwV1TnFTaQQ049iLMFMDBTXH0CpE2_O/view?usp=sharing).

• [**PolyCoder (0.4B).**](https://drive.google.com/file/d/1ImfNznQ7Ybl6gidkj9G_yh5Z_ueq-vbs/view?usp=sharing).

• [**CodeT5-small (60M)**](https://drive.google.com/file/d/1QwzvxJuWxsdcoHMGSNFwvzF7Zf_RDjcC/view?usp=sharing).

• [**CodeT5-base (220M)**](https://drive.google.com/file/d/18lmIO6I1GrXSbqemcEguuQBkOquGlflj/view?usp=sharing).

• [**CodeT5-large-ntp-py**](https://drive.google.com/file/d/1VuY5W1j9dQW2QbTSUV49yFV0__mAENjT/view?usp=sharing).




The code for model generation is here.

• [**CodeGen-mono (350M)**](https://drive.google.com/file/d/1gMBJftVAgQ5rkK7Ve-s8t3c7wn8PW1Cv/view?usp=sharing).

• [**PolyCoder (0.4B).**](https://drive.google.com/file/d/1bQgpkbcYi-W3Ro2C5LUUUjbXPxHnN9qN/view?usp=sharing).

• [**CodeT5-small (60M)**](https://drive.google.com/file/d/16DCKcTFtsJsmX3KK7Jx8j7hPpLLBg04E/view?usp=sharing).

• [**CodeT5-base (220M)**](https://drive.google.com/file/d/19ZPVBIBFHYtdQe8oUgScyDNRiGQvzzLZ/view?usp=sharing).

• [**CodeT5-large-ntp-py**](https://drive.google.com/file/d/1j9h-rAB8hhaLL2bh0BUY-PgxI-5flrT9/view?usp=sharing).

• [**ChatGPT**](https://drive.google.com/file/d/1x4VYlGhEifvtMn1GCp4FHc-4RFaileFp/view?usp=sharing).

