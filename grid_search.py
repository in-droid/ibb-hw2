import subprocess
# from ultralytics import YOLO
import os
import itertools
import datetime
import pickle
import subprocess

def train_model(lr, dropout, wd, optimizer):
    command = [
        "python3",  # or specify the path to your Python executable
        "train_script.py",  # replace with the actual script name
        "-lr", str(lr), 
        '-dp', str(dropout), 
        '-wd', str(wd), 
        '-o', optimizer]

    
    result = subprocess.run(command, capture_output=True, check=True)
    stdout = result.stdout.decode()  # Decode the stdout bytes to a string
    metric = stdout.split('\n')[-2].split(' ')[-1]
    # Process the stdout as needed
    # print(stdout)
    # print('*'*50)
    # print(stdout.split('\n')[-2])
    # print('*'*50)
    return float(metric)

if __name__ == '__main__':
        
    best_map = 0
    best_params = None
    all_metrics = {}
    # Move to the dir of this script
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    learning_rates = [0.0001, 0.001, 0.01]
    dropouts = [0.01, 0.1, 0.5]
    weight_decays = [0.0005, 0.005, 0.05]
    optimizers = ['SGD', 'Adam']


    param_combinations = list(itertools.product(
        learning_rates, dropouts, weight_decays, optimizers))

    for lr, dropout, wd, optimizer in param_combinations:
        mAp = train_model(lr, dropout, wd, optimizer)
        if mAp > best_map:
            best_map = mAp
            best_params = (lr, dropout, wd, optimizer)
    print("Training completed.")
    print("*"*50)
    print(f"Best mAP: {best_map}, Best params: {best_params}")

