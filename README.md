# Measuring Code Efficiency Optimization Capabilities with ACEOB

![论文流程图-2](https://github.com/CodeGeneration2/ACEOB/assets/95161813/0fed48ab-d004-4379-a79a-3ed70244e975)


## A B S T R A C T

As the gains from Moore’s Law gradually diminish, the performance and efficiency of software
have become increasingly critical. However, optimizing code efficiency remains a formidable
challenge, even for professional programmers. Surprisingly, research on code efficiency optimization is relatively sparse, and rigorously assessing models’ abilities to optimize code efficiency is particularly challenging. Furthermore, recent large-scale language models frequently
generate “slow positive”, failing to meet the time requirements of algorithms. In response to
these challenges, we first explored the “code patterns” encountered by models in the training
dataset, conducting an in-depth investigation and analysis of the content of the training dataset,
namely, human-written code. Second, we defined the task of code efficiency optimization and
introduced the Automatic Code Efficiency Optimization Benchmark (ACEOB), a standard for
evaluating code efficiency optimization capabilities. This benchmark contains 95,522 pairs of
efficient-inefficient codes submitted by human programmers to coding competitions. To our
knowledge, the ACEOB dataset is the first specifically dedicated to Python code efficiency
optimization. Then, to evaluate the model’s ability to optimize code efficiency, we proposed
two novel metrics: Isomorphic Optimal Comparison CodeBLEU (IOCCB) and Normalized
Performance Index (NPI), to assess the efficiency of the code generated by the models. Lastly,
we tested the performance of several state-of-the-art code generation models, such as PolyCoder
and CodeT5, after fine-tuning them on the ACEOB dataset. Additionally, we introduced the
NPI filter, and we found that the performance of all models improved after applying this filter.
However, we found that the current cutting-edge AI model—ChatGPT—does not perform ideally
in the task of code efficiency optimization. 


## Highlights

• ACEOB Dataset: We introduce the first benchmark dataset specifically designed for Python code efficiency
optimization tasks.

• IOCCB Score: We designed a variant of the CodeBLEU score to evaluate the model’s capability of generating
efficient code.

• NPI Score: We develop a novel evaluation metric that enables efficiency assessment of code without the necessity
of its execution.

• NPI Filter: We present a new type of filter intended for the ranking of code based on its efficiency.


## Example of IC2EC task. 

Each row corresponds to an efficient-inefficient code pair, consisting of inefficient code
(long running time) and efficient code (short running time).

![原始绘图数据6 drawio (1)](https://github.com/CodeGeneration2/ACEOB/assets/95161813/1b1aa5a4-e820-4c91-9052-6b28b02a168b)



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



## [**NPI Score Predictor**](https://drive.google.com/file/d/1XjWxYjBi6uLs5Pw-EAfzngJvGOaSglCS/view?usp=sharing).

When an IC is optimized into EC by a model, under ideal conditions (having the maximum/minimum code running
time that implements IC functionality, with the efficient code able to pass I/O unit tests). However, these ideal conditions are often unattainable. Despite the ACEOB dataset
having the maximum/minimum code running time and I/O unit tests, the use of NPI scores is still quite limited. To
enable the use of NPI scores to evaluate code efficiency for non-compilable code and other datasets, we specifically
trained an NPI score predictor to address this challenge.

The NPI score predictor is a model designed to forecast code NPI scores. We used CodeT5-base as the base
model and fine-tuned it on the ACEOB-NPI dataset. Unlike the IC2EC task, the “input features” of the predictor are the
code itself, while the “labels” used for gradient propagation are NPI scores. In the experimental section, we incorporate
the NPI score predictor as the NPI filter into the model, resulting in a substantial enhancement in model performance.

The parameters of the trained model are here（https://drive.google.com/file/d/1XjWxYjBi6uLs5Pw-EAfzngJvGOaSglCS/view?usp=sharing）.


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

