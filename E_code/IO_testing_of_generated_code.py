import os
from numpy import mean
import torch
import time
import json
import argparse
from tqdm import tqdm
import logging
import IO_test_module as test_util
IO_test_Log = None

def Receive_command_line_arguments():
    Command_line_parser = argparse.ArgumentParser()
    Command_line_parser.add_argument('--log_path', default='Log/IO_test.txt', type=str, required=False)
    Command_line_parameters = Command_line_parser.parse_args()
    return Command_line_parameters

def Create_log_file_functions(Command_line_parameters):
    Logs = logging.getLogger(__name__)
    Logs.setLevel(logging.INFO)
    Time_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    Log_file_writers = logging.FileHandler(filename=Command_line_parameters.log_path)
    Log_file_writers.setFormatter(Time_format)
    Log_file_writers.setLevel(logging.INFO)
    Logs.addHandler(Log_file_writers)
    Console = logging.StreamHandler()
    Console.setLevel(logging.DEBUG)
    Console.setFormatter(Time_format)
    Logs.addHandler(Console)
    return Logs

def IO_test_functions(Code_path, Training_set_or_test_set='NO', Number_of_rounds1=(- 1)):
    Total_list_of_compilation_rates = []
    Total_list_of_IO_test_pass_rates = []
    Generated_code_paths = os.listdir(Code_path)
    for (Code_Index, certain_code) in tqdm(enumerate(Generated_code_paths)):
        IO_test_Log.info(f'-------- IO_testing_of_generated_code: epochs {Number_of_rounds1} : {Training_set_or_test_set}: {certain_code} --------')
        IO_Path = certain_code.split('IO_Path_')[(- 1)][:(- 4)].replace('%', '/')
        with open(f'{Code_path}/{certain_code}', 'r', encoding='UTF-8') as f:
            Code = f.read()
        try:
            IO_test_result_list = test_util.run_test(IO_pair_paths=IO_Path, Code_String=Code, debug=False)
            certain_compilation_rate = 0
            Certain_IO_test_pass_rate = 0
            Number_of_IO_test_passes = 0
            if all(((data == (- 1)) for data in IO_test_result_list)):
                pass
            elif all(((data == False) for data in IO_test_result_list)):
                certain_compilation_rate = 1
            elif all(((data == True) for data in IO_test_result_list)):
                certain_compilation_rate = 1
                Certain_IO_test_pass_rate = 1
            elif all(((data == (- 2)) for data in IO_test_result_list)):
                certain_compilation_rate = 0
                Certain_IO_test_pass_rate = 0
            else:
                certain_compilation_rate = 1
                Number_of_IO_test_passes = IO_test_result_list.count(True)
                Certain_IO_test_pass_rate = (Number_of_IO_test_passes / len(IO_test_result_list))
        except:
            certain_compilation_rate = 0
            Certain_IO_test_pass_rate = 0
        Total_list_of_compilation_rates.append(certain_compilation_rate)
        Total_list_of_IO_test_pass_rates.append(Certain_IO_test_pass_rate)
        IO_test_Log.info(f'IO_testing, {Training_set_or_test_set}, {Code_Index}, Compilation rate:{certain_compilation_rate}, IO Pass Rate:{Certain_IO_test_pass_rate}, Code path:{Code_path}/{certain_code}')
    Average_compilation_rate = mean(Total_list_of_compilation_rates)
    Average_IO_Test_Pass_Rate = mean(Total_list_of_IO_test_pass_rates)
    Absolute_Correct_Number = Total_list_of_IO_test_pass_rates.count(1)
    with open(f'Generated_code/Round_{Number_of_rounds1}_prediction_code/Total_Average_Compile_Rate:{Average_compilation_rate:.3f}.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_compilation_rates))
    with open(f'Generated_code/Round_{Number_of_rounds1}_prediction_code/Total_Average_IO_Test_Pass_Rate:{Average_IO_Test_Pass_Rate:.3f},Number_of_code_passes:{Absolute_Correct_Number}.txt', 'w', encoding='UTF-8') as f:
        f.write(str(Total_list_of_IO_test_pass_rates))
    IO_test_Log.info(f'epochs:{Number_of_rounds1}, {Training_set_or_test_set}, Total_Average_Compile_Rate:{Average_compilation_rate:.3f}, Total_Average_IO_Test_Pass_Rate:{Average_IO_Test_Pass_Rate:.3f}, Number_of_code_passes:{Absolute_Correct_Number}')
    return (Average_compilation_rate, Average_IO_Test_Pass_Rate, Absolute_Correct_Number)

def main():
    Command_line_parameters = Receive_command_line_arguments()
    global IO_test_Log
    IO_test_Log = Create_log_file_functions(Command_line_parameters)
    IO_test_Log.info('#################################################### starting IO_test ##########################')
    Total_generated_code_paths = os.listdir(f'Generated_code')
    for round in tqdm(Total_generated_code_paths):
        if round.endswith('prediction_code'):
            time.sleep(1)
            Number_of_rounds1 = round.split('_prediction_code')[0].split('Round_')[1]
            (Test_set_average_compilation_rate, Test_set_average_IO_test_pass_rate, Absolute_number_of_correct_test_sets) = IO_test_functions(f'Generated_code/{round}/Test', 'Test', Number_of_rounds1)
            IO_test_Log.info(f'epochs:{Number_of_rounds1}, Average compilation rate of test set:{Test_set_average_compilation_rate}, Average IO test pass rate:{Test_set_average_IO_test_pass_rate}, Number of code passes:{Absolute_number_of_correct_test_sets}')
    IO_test_Log.info('######################################################## End IO_test ###########################')
if (__name__ == '__main__'):
    main()