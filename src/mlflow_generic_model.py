import mlflow
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import datetime
import uuid
import json
import base64
import os
from PIL import Image
from torchvision import transforms
import glob

class CrackDataset(Dataset):
    '''
    Crack Dataset for image segmentation tasks.
    Args:
        images (list): List of image file paths or numpy arrays.
        ground_truth (list): List of corresponding ground truth masks.
        transform (callable, optional): Optional transform to be applied on a sample.
        resize_size (tuple, optional): Desired output size (width, height) for resizing images and masks.
    Returns:
        torch.utils.data.Dataset: Dataset object for loading crack images and masks.
    '''
    def __init__(self, images_path, ground_truths_path = None, resize_size=(256, 256), sample_size=None):
        self.sample_size = sample_size
        image_exts = ["*.jpg", "*.jpeg", "*.png"]
        mask_exts = ["*.jpg", "*.jpeg", "*.png"]
        self.img_paths = sorted(
            [p for ext in image_exts for p in glob.glob(os.path.join(images_path, ext))]
        )
        if ground_truths_path is not None:
            self.mask_paths = sorted(
                [p for ext in mask_exts for p in glob.glob(os.path.join(ground_truths_path, ext))]
            )
        if sample_size is not None:
            self.img_paths = self.img_paths[:sample_size]
            self.mask_paths = self.mask_paths[:sample_size]
        self.resize_size = resize_size

    def resize_img(self, image, w, h):
        transform = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])
        return transform(image) 

    def resize_gt(self, gt, w, h):
        transform = transforms.Compose([
            transforms.Resize((w, h), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        return transform(gt)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")   # 3 channels
        if self.mask_paths is None:
            mask = None
        else:
            mask = Image.open(self.mask_paths[idx]).convert("L")   # 1 channel (grayscale)
        
        img = self.resize_img(img, self.resize_size[0], self.resize_size[1])
        if mask is not None:
            mask = self.resize_gt(mask, self.resize_size[0], self.resize_size[1])
            return img, mask

        return img, None
    

    
class PytorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_class, data_scaler, output_columns):
        self.model = model_class
        self.data_scaler = data_scaler
        self.output_columns = output_columns


    def load_context(self, context):
        self.model.load_state_dict(torch.load(context.artifacts["rul_model_path"]))
        self.model.eval()

    def set_data_loader(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=False)
        return data_loader

    def eval(self, data):
        data_loader = self.set_data_loader(data)
        outputs = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = data[0].to("cpu")
                output = self.model(data)
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).detach().numpy()
        return outputs

    def predict(self, model_input: pd.DataFrame):
        model_input = self.data_scaler.predict(model_input)
        output = self.eval(np.array([model_input.to_numpy()]))
        return pd.DataFrame(output, columns=self.output_columns).astype(float)

class AlarmGenerator(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def predict (self, model_input: pd.DataFrame):
        alarm_envelopes = []

        for _, row in model_input.iterrows():
            
            dm_envelope = {
                "ReceivedTopic": "",
                "CorrelationID": "",
                "ApiVersion": "v2",
                "RequestID": "",
                "ContentType": "application/json",
                "ErrorCode": 0,
                "Payload": "",
                "QueryParams": {}
            }

            presentDate = datetime.datetime.now()
            unix_timestamp = datetime.datetime.timestamp(presentDate) * 1000
            rul = row['RUL']
            if isinstance(rul, np.float64):
                rul = np.float32(rul)
            message = f'Current Remaining Useful Life {rul}'
            if row['RUL']<0.10:
                severity = "Critical"
            elif row['RUL']>0.10 and row['RUL']<0.30:
                severity = "Major"
            elif row['RUL']>0.30 and row['RUL']<0.50:
                severity = "Medium"
            else:
                severity = "Normal"
            alarm_data = {
                'message': message,
                'severity': severity,  # TBD
                'resolved': False,
                'description': "RUL in OBD2 data from vehicle",
                'job': "NA",
                'annotations': None
            }
            alarm_wrapper = {
                'alarm': alarm_data,
                'status': 'Online',
                'origin': int(unix_timestamp),
                'device_name': 'Volkswagen',  # hard coded
                'id': str(uuid.uuid4())
            }
            dm_envelope["Payload"] = base64.b64encode(json.dumps(alarm_wrapper).encode('utf-8'))

            alarm_envelopes.append(dm_envelope)

        if alarm_envelopes:
            alarm_df = pd.DataFrame(alarm_envelopes)
        else:
            alarm_df = pd.DataFrame(columns=['ReceivedTopic', 'CorrelationID', 'ApiVersion', 'RequestID', 'ContentType', 'ErrorCode','Payload','QueryParams'])

        return alarm_df
