# USE THIS FILE ONLY FOR THE FINAL EUCILNICA SUBMISSION!

from ultralytics import YOLO
import os, itertools, datetime

# Move to the dir of this script        
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

# Load the model, feel free to try other models
model = YOLO('runs/detect/train10/weights/best.pt')

metrics = model.val(data="ears_final_test.yaml")
print(metrics.results_dict['metrics/mAP50-95(B)'])