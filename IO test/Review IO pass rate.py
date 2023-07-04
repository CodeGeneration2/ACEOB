# coding=utf-8

import os

from tqdm import tqdm

import IO as test_util


def total_evaluation_function(evaluated_set_path, dataset_path):
    start_marker = True
    question_list = os.listdir(evaluated_set_path)
    for question_index in tqdm(question_list):
        various_code_pair_list = os.listdir(f"{evaluated_set_path}/{question_index}")
        for code_set in various_code_pair_list:
            code_list = os.listdir(f"{evaluated_set_path}/{question_index}/{code_set}")
            for code in code_list:

                if "newIO False newIO" in code or "newIO True newIO" in code:
                    continue

                if "IO False IO" in code:
                    new_name = code.replace(".txt", ",newIO False newIO.txt")
                    os.rename(f"{evaluated_set_path}/{question_index}/{code_set}/{code}",
                              f"{evaluated_set_path}/{question_index}/{code_set}/{new_name}")
                    continue

                if start_marker:
                    start_marker = False
                    new_name = code.replace(".txt", ",IO False IO.txt")
                    os.rename(f"{evaluated_set_path}/{question_index}/{code_set}/{code}",f"{evaluated_set_path}/{question_index}/{code_set}/{new_name}")
                    continue

                IO_path = f"{dataset_path}/{question_index}/Hide IO unit tests.json"

                result = evaluation_function(f"{evaluated_set_path}/{question_index}/{code_set}/{code}", IO_path)

                if result == 1:
                    new_name = code.replace(".txt", ",newIO True newIO.txt")
                else:
                    new_name = code.replace(".txt", ",newIO False newIO.txt")

                os.rename(f"{evaluated_set_path}/{question_index}/{code_set}/{code}", f"{evaluated_set_path}/{question_index}/{code_set}/{new_name}")




def evaluation_function(code_path,IO_path):
    print(code_path)
    # ---------------------------- read test case ---------------------#
    with open(f"{code_path}", 'r', encoding='UTF-8') as f:
        test_code = f.read()
    try:
        IO_test_result_list = test_util.run_test(prob_path=IO_path, test=test_code, debug=False)
        if all(x is True for x in IO_test_result_list):
            return 1
        else:
            return -1
    except:
        return -2




# ############################################################ entrance ####################################################
if __name__ == '__main__':
    # dataset_path = "../../43-ACEO/test"
    # generated_code_path = "./allcode"
    dataset_path = "../../test"
    generated_code_path = "../../reIO"
    generated_code_list = os.listdir(generated_code_path)
    for some_generated_code_set in tqdm(generated_code_list):
        total_evaluation_function(evaluated_set_path=f"{generated_code_path}/{some_generated_code_set}", dataset_path=dataset_path)
