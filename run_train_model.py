import os

import torch
import yaml
from ultralytics import YOLO  
from QtFusion.path import abs_path
device = "0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__': 
    workers = 1
    batch = 8

    data_name = "Tumor"
    data_path = abs_path(f'datasets/{data_name}/{data_name}.yaml', path_type='current') 
    unix_style_path = data_path.replace(os.sep, '/')

    directory_path = os.path.dirname(unix_style_path)
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    if 'path' in data:
        data['path'] = directory_path
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    model = YOLO(abs_path('./weights/yolov8n.pt'), task='detect')  
    results2 = model.train(  
        data=data_path,  
        device=device,  
        workers=workers,  
        imgsz=640,  
        epochs=120,  
        batch=batch,  
        name='train_v8_' + data_name  
    )

    model = YOLO(abs_path('./weights/yolov5nu.pt', path_type='current'), task='detect')  
    results = model.train(  
        data=data_path, 
        device=device, 
        workers=workers,  
        imgsz=640,  
        epochs=120,  
        batch=batch, 
        name='train_v5_' + data_name 
    )
