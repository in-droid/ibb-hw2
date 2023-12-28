import argparse
from pathlib import Path
from ultralytics import YOLO
import os, itertools, datetime
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument( '-dp' ,'--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer')
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
        optimizer=optimizer, 
        pretrained=True, 
        patience=3,
        plots=True,
        batch=32,
        val=True,
        augment=False,
        dropout=dropout,
        lr0=learning_rate,
        lrf=0.2,
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3,
        warmup_momentum=0.5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5
    )
    metrics = model.val() # It'll automatically evaluate the data you trained.
    Path(f'runs/detect/grid_search/train_{learning_rate}_{dropout}_{weight_decay}_{optimizer}').mkdir(parents=True, exist_ok=True)
    with open(f'runs/detect/grid_search/train_{learning_rate}_{dropout}_{weight_decay}_{optimizer}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics.results_dict, f)
    print("MAP:", metrics.results_dict['metrics/mAP50-95(B)'])
