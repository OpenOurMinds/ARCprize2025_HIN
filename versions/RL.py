from subprocess import Popen, PIPE, STDOUT

import torch
import os
import json
import glob
from pathlib import Path

from datetime import datetime
import pytz



import torch
import os
import time

# Set timezone
os.environ['TZ'] = 'America/Los_Angeles'
time.tzset()

# Get current time in new timezone
current_time = time.strftime('%Y-%m-%d %H:%M:%S')
print(current_time)

sample_path = '../input/arc-prize-2024/sample_submission.json'

def print_cuda_devices():
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
        
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s):")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

def is_in_slices_safe(number: int, slice_string: str) -> bool:
    """
    Safe version with error handling.
    """
    try:
        parts = slice_string.split(',')
        
        for part in parts:
            part = part.strip()  # Handle potential whitespace
            if ':' in part:
                start, end = map(int, part.split(':'))
                if start <= number <= end:
                    return True
            else:
                if number == int(part):
                    return True
        return False
    except ValueError:
        raise ValueError("Invalid slice format. Use format like '5:9,12,15:99'")
    except Exception as e:
        raise Exception(f"Error processing slices: {str(e)}")

def merge_with_sample(data_path, sample_path, sub_solver): 
    with open(sample_path,'r') as f:        
        sample = json.load(f) 

    # ...............................................................................
    with open(data_path,'r') as f:
        data = json.load(f)
        tasks_name = list(data.keys())
        tasks_file = list(data.values())

    for n in range(len(tasks_name)):
        task = tasks_file[n]
        t = tasks_name[n]
            
        for i in range(len(task['test'])): 
            # First check if task id exists
            if t not in sub_solver:
                sub_solver[t] = []
            
            # Ensure we have enough elements in the list
            while len(sub_solver[t]) <= i:
                sub_solver[t].append({})
            
            # Now check if attempt_1 exists or is empty
            if 'attempt_1' not in sub_solver[t][i] or not sub_solver[t][i]['attempt_1']:
                sub_solver[t][i]['attempt_1'] = sample[t][i]['attempt_1']
                
            # Same for attempt_2
            if 'attempt_2' not in sub_solver[t][i] or not sub_solver[t][i]['attempt_2']:
                sub_solver[t][i]['attempt_2'] = sample[t][i]['attempt_2']

    return sub_solver

def get_task_count():
    # Create the target datetime in PST
    pacific_tz = pytz.timezone('US/Pacific')
    target_date = pacific_tz.localize(datetime(2024, 11, 10, 11, 59))
    
    # Get current time in PST
    current_time = datetime.now(pacific_tz)
    
    # Compare and return appropriate value
    task_count = 9 if current_time < target_date else 2000
    return task_count

def mySystem(cmd):
    print(cmd)
    process = Popen(cmd, shell=True) # stdout=PIPE, stderr=STDOUT, 
    return process.wait() == 0 # do not assert here, to keep the program going

# Run the function
print_cuda_devices()

def find_transformer_model(directory):
    """
    Find the single Transformer*.pt file in the given directory and its subdirectories.
    
    Args:
        directory (str): Directory path to search in
    
    Returns:
        str: Full path to the found Transformer*.pt file
    
    Raises:
        AssertionError: If directory doesn't exist, no file found, or multiple files found
    """
    # Check if directory exists
    assert os.path.exists(directory), f"Directory '{directory}' does not exist"
    
    # Search for Transformer*.pt files
    pattern = os.path.join(directory, "**", "Transformer*.pt")
    matching_files = glob.glob(pattern, recursive=True)
    
    # Assert we found exactly one file
    assert len(matching_files) > 0, f"No Transformer*.pt file found in '{directory}'"
    assert len(matching_files) == 1, (
        f"Multiple Transformer*.pt files found in '{directory}': {matching_files}"
    )
    
    return matching_files[0]

def print_version_files():
    input_dir = "../input/"
    pattern = os.path.join(input_dir, "**", "__version__.txt")
    version_files = glob.glob(pattern, recursive=True)
    
    if not version_files:
        print("No __version__.txt files found")
        return
        
    for file_path in version_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            relative_path = os.path.relpath(file_path, input_dir)
            print(f"\n{relative_path}:")
            print(content)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

def is_tmp_writable():
    try:
        testfile = os.path.join('/tmp', 'write_test')
        with open(testfile, 'w') as f:
            f.write('test')
        os.remove(testfile)
        return True
    except (IOError, OSError):
        return False

assert is_tmp_writable()

# Display detailed GPU information
!nvidia-smi
!ls -R ../input

model_path = find_transformer_model("../input/transformer_model/")

# model_path = '../input/transformer_model/Transformer_best.pt'

print('\n\nmodel_path:', model_path)

# Add this line after line 164
print("\nVersion files found:")
print_version_files()

import os
import json
import time
    
# from shared import mySystem, merge_with_sample, get_task_count, is_in_slices_safe, sample_path

#######################################################################################
# Adapt ARC Prize 2024 files to work with Abstraction and Resoning Corpus 2020 rules ##
#######################################################################################

def adapt_2024_to_2020_rules(json_file_path, task_list_slices):
    # Load the JSON content
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create the 'test' directory
    output_dir = '../working/abstraction-and-reasoning-challenge/test'  
    os.makedirs(output_dir, exist_ok=True)

    # Split the JSON content into individual files
    for task_index, (task_id, task_data) in enumerate(data.items()):

        if is_in_slices_safe(task_index, task_list_slices):
            output_file_path = os.path.join(output_dir, f'{task_id}.json')
            with open(output_file_path, 'w') as output_file:
                json.dump(task_data, output_file, indent=4)

############################################
# Beginning of icecuber's original solution#
##########################################

def icecuber_solution():
    if open("../input/arc-solution-source-files-by-icecuber/version.txt").read().strip() == "671838222":
        print("Dataset has correct version")
    else:
        print("Dataset version not matching!")
        assert(0)
        
    mySystem("cp -r ../input/arc-solution-source-files-by-icecuber ./absres-c-files")
    mySystem("cd absres-c-files; make -j")
    mySystem("cd absres-c-files; python3 safe_run.py")
    mySystem("cp absres-c-files/submission_part.csv old_submission.csv")

# Function to translate from old submission format (csv) to new one (json)
def translate_submission(file_path):
    # Read the original submission file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # Skip the header line
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # Split predictions based on ' '
        
        # # Take only the first two predictions
        # if len(predictions) > 2:
        #     predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # Check if pred is not an empty string
                pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_dict = {
            "attempts": processed_predictions,
        }

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)
        else:
            submission_dict[task_id].append(attempt_dict)
    
    return submission_dict

def ice_main(test_path):
    print(f'ice_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    adapt_2024_to_2020_rules(test_path, f'0:{task_count - 1}')
    icecuber_solution()
    sub_dict = translate_submission('./old_submission.csv')
    # sub_dict = merge_with_sample(test_path, sample_path, sub_dict)

    with open('ice_submission_candidates.json', 'w') as file:
        json.dump(sub_dict, file, indent=4)

    print(f'ice_main Done @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

# if __name__ == "__main__":
#     # TODO do not use "evaluation" at submission time
#     ice_main(os.path.abspath('../input/arc-prize-2024/arc-agi_test_challenges.json'))


import time
import os
import json

# from shared import mySystem, get_task_count, merge_with_sample, sample_path

def soma_main(test_path):
    print(f'soma_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    mySystem("mkdir -p soma; cp -r ../input/python/* ./soma")
    mySystem(f"cd soma; python soma.py --test_path {test_path} --range 0:{task_count - 1}")
    print(f'soma_main Done. @ {time.strftime("%Y-%m-%d %H:%M:%S")}')

# if __name__ == "__main__":
#     soma_main(os.path.abspath('../input/arc-prize-2024/arc-agi_test_challenges.json'))

import torch
import os
import json
import time

# from shared import mySystem, merge_with_sample, get_task_count, model_path

def transformer_main(source):
    print(f'transformer_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    device_count = max(1, torch.cuda.device_count())

    mySystem("rm -rf transformer; cp -r ../input/transformer ./transformer")

    mySystem(f"cd transformer; python3 safe_run.py --checkpoint-path {os.path.abspath(model_path)} --source {source} --maximum-task-count {task_count} --process-count {device_count}")
        
    print(f'transformer_main Done. @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

# if __name__ == "__main__":
    
#     source = 'arc-agi_test'
#     test_path = os.path.abspath(f'../input/arc-prize-2024/{source}_challenges.json')

#     transformer_main(source)

#     with open('transformer/submission.json','r') as f:        
#         transformer_result = json.load(f) 

#     sample_path = '../input/arc-prize-2024/sample_submission.json'
#     sub_dict = merge_with_sample(test_path, sample_path, transformer_result)

#     with open('submission.json', 'w') as file:
#         json.dump(sub_dict, file, indent=4)

import json
import multiprocessing
import time
import os
import sys
import subprocess

# from transformer import transformer_main
# from soma import soma_main  # Assuming there's a soma_main function
# from icecuber import ice_main  # Assuming there's an icecuber_main function
# from shared import mySystem

def append_if_not_exist(attempts_dict, new_attempts, score):
    for idx, attempt in enumerate(new_attempts):
        # Convert attempt to tuple of tuples for hashability
        attempt_key = tuple(map(tuple, attempt))
        # Calculate weighted score based on position
        weighted_score = score * ((1 / 8) ** idx)
        # Add or accumulate the score
        attempts_dict[attempt_key] = attempts_dict.get(attempt_key, 0) + weighted_score

        print(f'+{weighted_score} ({score}[{idx}]) -> {attempts_dict[attempt_key]} for {attempt_key}')
    
    return attempts_dict

def build_top_2_attempts(t, i, *, soma, ice, transformer):
    attempts_dict = {}
    try:
        append_if_not_exist(attempts_dict, soma[t][i]['attempts'], 0.3)
    except:
        pass

    try:
        append_if_not_exist(attempts_dict, ice[t][i]['attempts'], 0.18)
    except:
        pass

    try:
        append_if_not_exist(attempts_dict, transformer[t][i]['attempts'], 0.20)
    except:
        pass

    sorted_items = sorted(attempts_dict.items(), key=lambda x: x[1], reverse=True)

    for index, item in enumerate(sorted_items):
        print(f'{t}[{i}][{index}]', item)

    # Convert back to list and sort by votes
    sorted_attempts = [list(map(list, k)) for k, v in 
                        sorted_items]

    if len(sorted_attempts) >= 2:
        answer = {'attempt_1': sorted_attempts[0], 'attempt_2': sorted_attempts[1]}
    elif len(sorted_attempts) >= 1:
        answer = {'attempt_1': sorted_attempts[0], 'attempt_2': [[0]]}
    else:
        answer = {'attempt_1': [[0]], 'attempt_2': [[0]]}

    return answer

def merge_all_with_sample(data_path, transformer_submission, soma_submission, ice_submission): 
    with open(data_path,'r') as f:
        data = json.load(f)
        sorted_items = sorted(data.items(), key=lambda x: str(x[1]))
        tasks_name = [item[0] for item in sorted_items]
        tasks_file = [item[1] for item in sorted_items]

    sub_solver = {}

    for n in range(len(tasks_name)):
        task = tasks_file[n]
        t = tasks_name[n]
        sub_solver[t] = []
            
        for i in range(len(task['test'])):
            answer = build_top_2_attempts(t, i, soma = soma_submission, ice = ice_submission, transformer = transformer_submission)

            sub_solver[t].append(answer)

    return sub_solver

def run_with_output_redirect(target, args, output_file):
    # Create a copy of the original stdout and stderr
    old_stdout = os.dup(sys.stdout.fileno())
    old_stderr = os.dup(sys.stderr.fileno())
    
    # Fork tee process
    tee_stdout = subprocess.Popen(
        ['tee', output_file],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        text=True
    )
    
    try:
        # Redirect stdout and stderr to the tee process
        os.dup2(tee_stdout.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee_stdout.stdin.fileno(), sys.stderr.fileno())
        
        # Run the target function
        target(*args)
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout, sys.stdout.fileno())
        os.dup2(old_stderr, sys.stderr.fileno())
        
        # Close duplicated file descriptors
        os.close(old_stdout)
        os.close(old_stderr)
        
        # Ensure tee process is terminated
        tee_stdout.stdin.close()
        tee_stdout.wait()

def ensemble_main():
    source = 'arc-agi_test'
    test_path = os.path.abspath(f'../input/arc-prize-2024/{source}_challenges.json')

    print(f'Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Create processes for each main function with tee output
    # processes = [
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(transformer_main, (test_path, source), 'transformer_output.log')
    #     ),
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(soma_main, (test_path,), 'soma_output.log')
    #     ),
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(ice_main, (test_path,), 'ice_output.log')
    #     )
    # ]
    
    processes = [
        multiprocessing.Process(
            target=transformer_main,
            args=(source, )
        ),
        multiprocessing.Process(
            target=soma_main,
            args=(test_path,)
        ),
        multiprocessing.Process(
            target=ice_main,
            args=(test_path,)
        )
    ]    
    # Start all processes
    for p in processes:
        p.start()
    
    print('All process started.')
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f'All process ended. @ {time.strftime("%Y-%m-%d %H:%M:%S")} All parallel processes completed, running ensemble...')

    with open('transformer/submission_candidates.json','r') as f:
        transformer_result = json.load(f) 

    with open('soma/submission_candidates.json', 'r') as file:
        soma_submission = json.load(file)

    with open('ice_submission_candidates.json', 'r') as file:
        ice_submission = json.load(file)

    submission = merge_all_with_sample(test_path, transformer_result, soma_submission, ice_submission)
    
    with open('submission.json', 'w') as file:
        json.dump(submission, file, indent=4)

    # otherwise, kaggle cannot find our answer!
    mySystem("tar -czf abs_store.tar.gz absres-c-files/store")
    mySystem("cd absres-c-files; find . -maxdepth 1 -type d -not -path . -exec rm -r {} +")

    mySystem("rm -r abstraction-and-reasoning-challenge")

    mySystem("tar -czf transformer_store.tar.gz transformer/store")
    mySystem("tar -czf transformer_submission.tar.gz transformer/submission")

    mySystem("cd transformer; find . -maxdepth 1 -type d -not -path . -exec rm -r {} +")

if __name__ == "__main__":
    ensemble_main()