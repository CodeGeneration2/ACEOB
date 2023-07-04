# coding=utf-8
"""
Dataset to be used for APPS Training
"""
import time

import os

from tqdm import tqdm

from Single_generation import generate_code


# =========================================== Initialization function (import data from local) ==================#
def init_function(testset_path):
    historical_timestamp = 0
    dataset_list = os.listdir(f"{testset_path}")
    # dataset_list = dataset_list[:20]
    for question_index in tqdm(dataset_list):
        print(f"Start question {question_index}")
        # ================================================== Feature =======================================#
        question_content_list = os.listdir(f"{testset_path}/{question_index}")
        for some_content in tqdm(question_content_list):
            if "efficient-inefficient code pair" in some_content:
                # ============================================================================================
                generated_code_store_path = f"F:/ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted/generated code and trained models/super-computing-cloud-generated-all-codes-some-have-NPI-scores/ChatGPT-original-processing/{question_index}/{some_content}"
                if os.path.exists(generated_code_store_path):
                    if len(os.listdir(f"{generated_code_store_path}")) != 0:
                        continue
                else:
                    os.makedirs(generated_code_store_path)
                # ================================================== Feature =======================================#
                efficient_inefficient_code_list = os.listdir(f"{testset_path}/{question_index}/{some_content}")
                # ========================================== Import feature set for some record ==================#
                for some_code in efficient_inefficient_code_list:
                    if "inefficient" in some_code:
                        with open(f"{testset_path}/{question_index}/{some_content}/{some_code}", 'r', encoding='UTF-8') as f:
                            slow_code = f.read()

                timestamp = time.time()

                if timestamp < historical_timestamp + 21:
                    time.sleep(round(historical_timestamp + 21 - timestamp))
                try:
                    efficient_code = generate_code(slow_code)
                except:
                    time.sleep(20)
                    continue

                historical_timestamp = time.time()

                # ---------------------------- Save the record file -------------------------------------------------#
                with open(f"{generated_code_store_path}/0.txt", 'w', encoding='UTF-8') as f:
                    f.write(efficient_code)



# ==================================================================================#
# ==================================================================================#
# ==================================================================================#
if __name__ == '__main__':
    testset_path = "F:/ACEO-current-code-efficiency-optimization-dataset-paper1-not-submitted/dataset/45-ACEO/test"
    init_function(testset_path)
