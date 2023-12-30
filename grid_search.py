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
        # '--augment' if augment else '',
        '--hsv_h', str(augment_dict.get('hsv_h', 0)),
        '--hsv_s', str(augment_dict.get('hsv_s', 0)),
        '--hsv_v', str(augment_dict.get('hsv_v', 0)),
        '--degrees', str(augment_dict.get('degrees', 0)),
        '--mixup', str(augment_dict.get('mixup', 0)),
        '--mosaic', str(augment_dict.get('mosaic', 0)),
        '--scale', str(augment_dict.get('scale', 0))
        ]
    if augment:
        command.append('--augment')

    
    result = subprocess.run(command, capture_output=True, check=True)
    stdout = result.stdout.decode()  # Decode the stdout bytes to a string
    metric = stdout.split('\n')[-2].split(' ')[-1]
    return float(metric)

if __name__ == '__main__':
        
    best_map = 0
    best_map_augment = 0
    best_params = None
    best_augment_params = None

    all_metrics = {}
    # Move to the dir of this script
    dname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dname)

    # learning_rates = [0.0001, 0.001, 0.01]
    learning_rates = [0.0001, 0.001, 0.01]
    dropouts = [0.01, 0.1, 0.5]
    weight_decays = [0.0005, 0.005, 0.05]
    optimizers = ['SGD', 'Adam']


    param_combinations = list(itertools.product(learning_rates,
                                                dropouts,
                                                weight_decays,
                                                optimizers))
    

    # learning_rate = 0.00
    # dropout = 0.01
    # wd = 0.005
    # optimizer = 'Adam'
    # best metrics from the first step

    hsv_h_params = [0.015, 0.1, 0.2]
    hsv_s_params = [0.1, 0.7]
    hsv_v_params = [0.1, 0.4]
    degrees_params = [0, 45]
    mixup_params = [0, 0.2, 0.5]
    mosaic = [1.0]
    scale = [0.5]
    

    augmentation_param_combinations = list(itertools.product(
        hsv_h_params, 
        hsv_s_params, 
        hsv_v_params, 
        degrees_params,
        mixup_params,
        mosaic,
        scale
        ))
    
    print(f'Number of param combinations: {len(param_combinations)}')
    for lr, dropout, wd, optimizer in tqdm(param_combinations):
        mAp = train_model(lr, dropout, wd, optimizer, augment_dict=None)
        if mAp > best_map:
            best_map = mAp
            best_params = (lr, dropout, wd, optimizer)

    print("Grid search for params completed.")
    print("*"*50)

    named_params = zip(['lr', 'dropout', 'wd', 'optimizer'], best_params)
    named_params = ', '.join([f'{name}: {value}' for name, value in named_params])
    print(f"Best mAP: {best_map}, Best params: {named_params}")
    best_lr, best_dropout, best_wd, best_optimizer = best_params


    print('*'*50)
    print('Grid search for augmentation started.')
    print('*'*50)
    print(f'Number of combinations: {len(param_combinations)}')
    print("| ".join(['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'mixup', " mAP"]))
    for hsv_h, hsv_s, hsv_v, degrees, mixup, scale_, mosaic_ in tqdm(augmentation_param_combinations):
        mAp = train_model(best_lr, best_dropout, best_wd, best_optimizer, augment_dict={
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'hsv_v': hsv_v,
            'degrees': degrees,
            'mixup': mixup,
            'scale': scale_,
            'mosaic': mosaic_
        }
        )
        if mAp > best_map_augment:
            best_map_augment = mAp
            best_params_augment = (hsv_h, hsv_s, hsv_v, degrees, mixup)
    print('Training completed.')
    print('*'*50)
    best_params_with_names = zip(
        ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'mixup'],
        best_params_augment
    )

    params_string_augment = ', '.join([f'{name}: {value}' for name, value in best_params_with_names])

    print(f'Best mAP: {best_map}')
    print(f'Best params: {named_params}')
    print(f'Best mAP with augmentation: {best_map_augment}')
    print(f'Best params with augmentation: {params_string_augment}')