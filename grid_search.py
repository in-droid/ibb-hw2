import subprocess
from tqdm import tqdm
# from ultralytics import YOLO
import os
import itertools
import datetime
import pickle


def train_model(lr, dropout, wd, optimizer, augment_dict=None):
    if augment_dict is None:
        augment_dict = {}
        augment = False
    else:
        augment = True

    command = [
        "python3",  # or specify the path to your Python executable
        "train_script.py",  # replace with the actual script name
        "-lr", str(lr), 
        '-dp', str(dropout), 
        '-wd', str(wd), 
        '-o', optimizer,
        '--augment' if augment else '',
        '--hsv_h', str(augment_dict.get('hsv_h', 0)),
        '--hsv_s', str(augment_dict.get('hsv_s', 0)),
        '--hsv_v', str(augment_dict.get('hsv_v', 0)),
        '--degrees', str(augment_dict.get('degrees', 0)),
        '--mixup', str(augment_dict.get('mixup', 0))
        ]

    
    result = subprocess.run(command, capture_output=True, check=True)
    stdout = result.stdout.decode()  # Decode the stdout bytes to a string
    metric = stdout.split('\n')[-2].split(' ')[-1]
    return float(metric)

if __name__ == '__main__':
        
    best_map = 0
    best_params = None
    all_metrics = {}
    # Move to the dir of this script
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    # learning_rates = [0.0001, 0.001, 0.01]
    learning_rate = 0.00
    dropout = 0.01
    wd = 0.005
    optimizer = 'Adam'
    # best metrics from the first step

    hsv_h_params = [0.015, 0.1, 0.2]
    hsv_s_params = [0.1, 0.7]
    hsv_v_params = [0.1, 0.4]
    degrees_params = [0, 45]
    mixup_params = [0, 0.2, 0.5]
    

    param_combinations = list(itertools.product(
        hsv_h_params, 
        hsv_s_params, 
        hsv_v_params, 
        degrees_params,
        mixup_params
        ))


    print('Grid search started.')
    print('*'*50)
    print(f'Number of combinations: {len(param_combinations)}')
    print("| ".join(['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'mixup', " mAP"]))
    for hsv_h, hsv_s, hsv_v, degrees, mixup in tqdm(param_combinations):
        mAp = train_model(learning_rate, dropout, wd, optimizer, augment_dict={
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'hsv_v': hsv_v,
            'degrees': degrees,
            'mixup': mixup
        }
        )
        print(f'{hsv_h} | {hsv_s} | {hsv_v} | {degrees} | {mixup} || {mAp: .3f}')
        if mAp > best_map:
            best_map = mAp
            best_params = (hsv_h, hsv_s, hsv_v, degrees, mixup)
    print('Training completed.')
    print('*'*50)
    best_params_with_names = zip(
        ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'mixup'],
        best_params
    )

    params_string = ', '.join([f'{name}: {value}' for name, value in best_params_with_names])

    print(f'Best mAP: {best_map}, Best params: {params_string}')

    # for lr, dropout, wd, optimizer in tqdm(param_combinations):
    #     mAp = train_model(lr, dropout, wd, optimizer)
    #     if mAp > best_map:
    #         best_map = mAp
    #         best_params = (lr, dropout, wd, optimizer)
    # print("Training completed.")
    # print("*"*50)
    # print(f"Best mAP: {best_map}, Best params: {best_params}")

