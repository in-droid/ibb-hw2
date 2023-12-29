import argparse
from pathlib import Path
from ultralytics import YOLO
import os, itertools, datetime
import pickle

OUTPUT_PATH = 'runs/detect/grid_search'

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument( '-dp' ,'--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('-a', '--augment', action='store_true', help='Augment data')
    parser.add_argument('--hsv_h', type=float, default=0, help='HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0, help='HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0, help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0, help='Rotation augmentation')
    parser.add_argument('--mixup', type=float, default=0, help='Mixup augmentation')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    learning_rate = args.learning_rate
    dropout = args.dropout
    weight_decay = args.weight_decay
    optimizer = args.optimizer
    
    model = YOLO("yolov8n.pt")
    model.train(
        data="ears.yaml", 
        epochs=20, 
        epochs=20, 
        optimizer=optimizer, 
        pretrained=True, 
        patience=3,
        plots=True,
        batch=32,
        val=True,
        augment=args.augment,
        dropout=dropout,
        lr0=learning_rate,
        lrf=0.2,
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3,
        warmup_momentum=0.5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        mixup=args.mixup
    )
    metrics = model.val() # It'll automatically evaluate the data you trained.
    if args.augment:
        output_path = os.path.join(OUTPUT_PATH, 'augment', f'hsv_h_{args.hsv_h}_hsv_s_{args.hsv_s}_hsv_v_{args.hsv_v}_degrees_{args.degrees}_mixup_{args.mixup}')
    else:
        output_path = os.path.join(OUTPUT_PATH, f'train_{learning_rate}_{dropout}_{weight_decay}_{optimizer}')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics.results_dict, f)  
    print("MAP:", metrics.results_dict['metrics/mAP50-95(B)'])
    