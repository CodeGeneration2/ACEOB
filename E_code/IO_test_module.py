import argparse
import json
import os
import sys
import io
import faulthandler
from datetime import datetime
import signal
import numpy as np
from io import StringIO
from typing import get_type_hints
from typing import List, Tuple
from unittest.mock import patch, mock_open
from pyext import RuntimeModule
from enum import Enum

class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    print('alarm went off')
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 5

class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = (lambda x: 1)
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def parse_args():
    parser = argparse.ArgumentParser(description='Utility for testing code generation.')
    parser.add_argument('-v', '--verbosity-level', action='store', type=int, help='')
    parser.add_argument('-s', '--source', type=str, default='leetcode', choices=['leetcode', 'atcoder', 'codewars'], help='which data source to gather from.')
    parser.add_argument('-d', '--data', type=str, default='question', choices=['question', 'q', 'solutions', 'sol', 's', 'starter', 'tests', 't'], help='which type of data to receive.')
    parser.add_argument('-n', '--number', type=int, default=0, help='which problem to query.')
    args = parser.parse_args()
    return args

def get_valid_problems(data_dir='leetcode'):
    if (data_dir == 'leetcode'):
        root = os.path.join(args.source, 'data')
    elif (data_dir == 'atcoder'):
        pass
    root = os.path.join(data_dir, 'data')
    if os.path.exists(os.path.join(data_dir, 'valid_problems.json')):
        with open(os.path.join(data_dir, 'valid_problems.json'), 'r') as f:
            return json.load(f)
    tmp = os.listdir(root)
    valid_probs = []
    for folder in tmp:
        prob_path = os.path.join(root, folder)
        files = os.listdir(prob_path)
        if (('input_output.json' in files) or ('sols.json' in files)):
            valid_probs.append(prob_path)
    valid_probs = sorted(valid_probs)
    return valid_probs

def get_question(problem_list, prob_index):
    root = problem_list[prob_index]
    if os.path.exists(os.path.join(root, 'question.txt')):
        with open(os.path.join(root, 'question.txt')) as f:
            question = f.readlines()
    else:
        print('question prompt not found')
        question = ''
    question = ''.join(question)
    return question

def get_solutions(problem_list, prob_index):
    root = problem_list[prob_index]
    if os.path.exists(os.path.join(root, 'solutions.json')):
        with open(os.path.join(root, 'solutions.json')) as f:
            sols = json.load(f)
    return sols

def run_test(IO_pair_paths: str=None, problem_list: List[str]=None, prob_index: int=None, Code_String: str=None, debug: bool=False):
    if ((IO_pair_paths is None) and (problem_list is None)):
        print('please provide either IO_pair_paths or problem_list')
        exit()
    if debug:
        print(f'start = {datetime.now().time()}')
    if (IO_pair_paths is not None):
        IO_pair_dictionary = IO_pair_paths
    elif (problem_list is not None):
        IO_pair_dictionary = problem_list[prob_index]
    if os.path.exists(IO_pair_dictionary):
        with open(IO_pair_dictionary, 'r', encoding='UTF-8') as f:
            in_outs = f.read()
            in_outs = eval(in_outs)
            if debug:
                print(f"Code_String cases json = {in_outs['inputs']} {in_outs['outputs']}")
            if (in_outs.get('fn_name') is None):
                which_type = CODE_TYPE.standard_input
                method_name = None
            else:
                which_type = CODE_TYPE.call_based
                method_name = in_outs['fn_name']
    if debug:
        print(f'loaded json = {datetime.now().time()}')
    if (Code_String is None):
        return in_outs
    elif (Code_String is not None):
        results = []
        sol = 'import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n'
        if debug:
            print(f'loading Code_String code = {datetime.now().time()}')
        if (which_type == CODE_TYPE.call_based):
            sol += Code_String
            if debug:
                print(f'sol = {sol}')
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string('tmp_sol', '', sol)
                if ('class Solution' not in Code_String):
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f'type 0 compilation error = {e}')
                results.append((- 2))
                return results
            signal.alarm(0)
        elif (which_type == CODE_TYPE.standard_input):
            tmp_test = Code_String.split('\n')
            new_test = []
            for x in tmp_test:
                if ((not x.startswith('from ')) and (not x.startswith('import '))):
                    new_test.append((('\t' + x) + '\n'))
                else:
                    new_test.append((x + '\n'))
            tmp_test = new_test
            new_test = ''
            started = False
            for i in tmp_test:
                if (i.startswith('\t') and (not started)):
                    new_test += 'stdin = sys.stdin\nstdout = sys.stdout\n'
                    new_test += 'def code():\n'
                    new_test += i
                    started = True
                elif (started and (i.startswith('from ') or i.startswith('import '))):
                    new_test += ('\t' + i)
                else:
                    new_test += i
            tmp_test = new_test
            sol += tmp_test
            if debug:
                print(f'sol = {sol}')
            method_name = 'code'
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string('tmp_sol', '', sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f'type 1 compilation error = {e}')
                results.append((- 2))
                return results
            signal.alarm(0)
        if debug:
            print(f'get method = {datetime.now().time()}')
        try:
            method = getattr(tmp, method_name)
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f'unable to get function error = {e}')
            return results
        for (index, inputs) in enumerate(in_outs['inputs']):
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for (k, v) in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs['outputs'][index], dict):
                    in_outs['outputs'][index] = [{int(k): v for (k, v) in in_outs['outputs'][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs['outputs'][index][0], dict):
                    in_outs['outputs'][index] = [{int(k): v for (k, v) in in_outs['outputs'][index][0].items()}]
            except:
                True
            if debug:
                print(f'time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}')
            if (which_type == CODE_TYPE.call_based):
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)
                    if isinstance(output, tuple):
                        output = list(output)
                    tmp_result = (output == in_outs['outputs'][index])
                    if (isinstance(in_outs['outputs'][index], list) and in_outs['outputs'][index]):
                        tmp_result = (tmp_result or (output == in_outs['outputs'][index][0]))
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = (tmp_result or ([list(x) for x in output] == in_outs['outputs'][index][0]))
                    except:
                        True
                    results.append(tmp_result)
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    print(f'Standard input runtime error or time limit exceeded error = {e}')
                    results.append((- 1))
                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(f"outputs = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
            elif (which_type == CODE_TYPE.standard_input):
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False
                if isinstance(inputs, list):
                    inputs = '\n'.join(inputs)
                if isinstance(in_outs['outputs'][index], list):
                    in_outs['outputs'][index] = '\n'.join(in_outs['outputs'][index])
                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        signal.alarm(0)
                        print(f'Call-based runtime error or time limit exceeded error = {repr(e)}{e}')
                        results.append((- 1))
                    signal.alarm(0)
                if (not passed):
                    if debug:
                        nl = '\n'
                        if (not isinstance(inputs, list)):
                            print(f"not passed output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
                        else:
                            print(f"not passed output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
                    continue
                if (passed and debug):
                    print(f"==> output = {output}, Code_String outputs = {in_outs['outputs'][index]}")
                if custom_compare_(output, in_outs['outputs'][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue
                if isinstance(output, tuple):
                    output = list(output)
                tmp_result = False
                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = (tmp_result or (output == in_outs['outputs'][index]))
                        if isinstance(output[0], str):
                            tmp_result = (tmp_result or ([e.strip() for e in output] == in_outs['outputs'][index]))
                except Exception as e:
                    print(f'Failed check1 exception = {e}')
                    pass
                if (tmp_result == True):
                    results.append(tmp_result)
                    continue
                if isinstance(in_outs['outputs'][index], list):
                    for (tmp_index, i) in enumerate(in_outs['outputs'][index]):
                        in_outs['outputs'][index][tmp_index] = i.split('\n')
                        in_outs['outputs'][index][tmp_index] = [x.strip() for x in in_outs['outputs'][index][tmp_index] if x]
                else:
                    in_outs['outputs'][index] = in_outs['outputs'][index].split('\n')
                    in_outs['outputs'][index] = list(filter(len, in_outs['outputs'][index]))
                    in_outs['outputs'][index] = list(map((lambda x: x.strip()), in_outs['outputs'][index]))
                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = (tmp_result or (output == in_outs['outputs'][index]))
                except Exception as e:
                    print(f'Failed check2 exception = {e}')
                    pass
                if (tmp_result == True):
                    results.append(tmp_result)
                    continue
                if isinstance(output, list):
                    output = list(filter(len, output))
                if debug:
                    nl = '\n'
                    if (not isinstance(inputs, list)):
                        print(f"output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
                    else:
                        print(f"output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
                if (tmp_result == True):
                    results.append(tmp_result)
                    continue
                try:
                    tmp_result = (output == [in_outs['outputs'][index]])
                    if isinstance(in_outs['outputs'][index], list):
                        tmp_result = (tmp_result or (output == in_outs['outputs'][index]))
                except Exception as e:
                    print(f'Failed check3 exception = {e}')
                    pass
                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs['outputs'][index]]
                    tmp_result = (tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)))
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                        tmp_result = (tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)))
                except Exception as e:
                    pass
                if (tmp_result == True):
                    results.append(tmp_result)
                    continue
                if isinstance(in_outs['outputs'][index], list):
                    for (tmp_index, i) in enumerate(in_outs['outputs'][index]):
                        in_outs['outputs'][index][tmp_index] = set(i.split())
                else:
                    in_outs['outputs'][index] = set(in_outs['outputs'][index].split())
                try:
                    tmp_result = (output == in_outs['outputs'][index])
                except Exception as e:
                    print(f'Failed check4 exception = {e}')
                    continue
                if (tmp_result == True):
                    results.append(tmp_result)
                    continue
                if isinstance(output, list):
                    for (tmp_index, i) in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for (tmp_index, i) in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)
                try:
                    tmp_result = (set((frozenset(s) for s in output)) == set((frozenset(s) for s in in_outs['outputs'][index])))
                except Exception as e:
                    print(f'Failed check5 exception = {e}')
                try:
                    tmp_result = (tmp_result or (set((frozenset((round(float(t), 3) for t in s)) for s in output)) == set((frozenset((round(float(t), 3) for t in s)) for s in in_outs['outputs'][index]))))
                except Exception as e:
                    print(f'Failed check6 exception = {e}')
                if ((tmp_result == True) and debug):
                    print('PASSED')
                results.append(tmp_result)
                if debug:
                    nl = '\n'
                    if (not isinstance(inputs, list)):
                        print(f"output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
                    else:
                        print(f"output = {output}, Code_String outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {(output == [in_outs['outputs'][index]])}")
    return results

def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = '\n'.join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True
    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = '\n'.join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True
    return False

def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return (s1 == s2)

def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)
    inputs_line_iterator = iter(inputs.split('\n'))

    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', (lambda *args: next(inputs_line_iterator)))
    @patch('sys.stdin.readlines', (lambda *args: inputs.split('\n')))
    @patch('sys.stdin.read', (lambda *args: inputs))
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass
    return _inner_call_method(method)

def main(args):
    print(args)
    problem_list = sorted(get_valid_problems(args.source))
    print(f'number of problems = {len(problem_list)}')
    prob_index = args.number
    print(f'problem is {problem_list[prob_index]}')
    assert (prob_index < len(problem_list))
    if ((args.data == 'q') or (args.data == 'question')):
        tmp = get_question(problem_list, prob_index)
        print('q', tmp)
    elif (args.data in ['solutions', 'sol', 's']):
        tmp = get_solutions(problem_list, prob_index)
        print('sol', tmp)
    elif (args.data == 'starter'):
        tmp = get_starter(problem_list, prob_index)
        print('starter', tmp)
    elif (args.data in ['Code_String', 't']):
        sols = get_solutions(problem_list, prob_index)
        tmp = run_test(problem_list, prob_index, Code_String=sols[0])
        print('results = ', tmp)
        print('-2 = compile error, -1 is runtime error, False failed Code_String, True passed Code_String')
if (__name__ == '__main__'):
    args = parse_args()
    main(args)