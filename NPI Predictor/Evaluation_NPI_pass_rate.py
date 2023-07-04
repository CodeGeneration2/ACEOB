# coding=utf-8

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from generate_prompt_NPI import generate_NPI

from tqdm import tqdm


def total_evaluation_function(evaluated_set_path):
    problem_list = os.listdir(evaluated_set_path)
    for problem_index in tqdm(problem_list):
        various_efficiency_code_pairs_list = os.listdir(f"{evaluated_set_path}/{problem_index}")
        for code_list_name in various_efficiency_code_pairs_list:
            code_list = os.listdir(f"{evaluated_set_path}/{problem_index}/{code_list_name}")
            for a_code in code_list:
                # ---------------------------- Save record file -----------------------------------------------------------#
                with open(f"{evaluated_set_path}/{problem_index}/{code_list_name}/{a_code}", 'r', encoding='UTF-8') as f:
                    code_text = f.read()

                if "1newNPI1" in a_code:
                    continue

                # try:
                NPI = generate_NPI(code_text)
                try:
                    NPI = float(NPI)/100
                except:
                    print(f"Error: NPI: {NPI}")
                    NPI = 0
                # print("---",NPI)

                # except:
                #     print(f"！！！！！！！！！！！{evaluated_set_path}/{problem_index}/{a_code_set}/{a_code}")
                #     continue

                new_name = a_code.replace(".txt", f",1newNPI1 {NPI} 1newNPI1.txt")

                os.rename(f"{evaluated_set_path}/{problem_index}/{code_list_name}/{a_code}",
                          f"{evaluated_set_path}/{problem_index}/{code_list_name}/{new_name}")

                # if len(NPI) == 1:
                #     new_name = a_code.replace(".txt", f",NPI True NPI.txt")
                # else:
                #     new_name = a_code.replace(".txt", f",IO {NPI} IO.txt")
                #
                # os.rename(f"{evaluated_set_path}/{problem_index}/{a_code_set}/{a_code}", f"{evaluated_set_path}/{problem_index}/{a_code_set}/{new_name}")


# ############################################################ Entry ####################################################
if __name__ == '__main__':

    generated_code_path = "../../server-completed IO and IOCCB and NPI"

    # total_evaluation_function(evaluated_set_path="../../63-NPI training set-temporary/test")

    generated_code_list = os.listdir(generated_code_path)
    for a_generated_code_set in tqdm(generated_code_list):
        total_evaluation_function(evaluated_set_path=f"{generated_code_path}/{a_generated_code_set}")
