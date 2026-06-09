
import mlflow

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import os
import glob
from networks import *

import tqdm
import numpy as np
import cv2
import json
import uuid

import pandas as pd
import matplotlib.pyplot as plt

def soft_erode(x):
    # 3x3 min-pool approximation via -maxpool(-x)
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

def soft_skeletonize(p, iters=3):
    # p in [0,1]; iterative soft skeletonization
    skel = torch.zeros_like(p)
    img = p
    for _ in range(iters):
        er = soft_erode(img)
        op = F.relu(img - er)
        skel = torch.maximum(skel, op)
        img = er
    return skel

def soft_clDice(pred_prob, target, iters=3, eps=1e-6):
    # pred_prob, target: (B,1,H,W) in [0,1]
    S_p = soft_skeletonize(pred_prob, iters)
    S_t = soft_skeletonize(target, iters)
    tprec = (S_p * target).sum(dim=(1,2,3)) / (S_p.sum(dim=(1,2,3)) + eps)
    tsens = (S_t * pred_prob).sum(dim=(1,2,3)) / (S_t.sum(dim=(1,2,3)) + eps)
    cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
    return cldice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=pos_weight)  # or BCEWithLogitsLoss for binary
        self.gamma = gamma

    def dice_loss(self, pred, target, smooth=1.0):
        if pred.size(1) == 1:
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1)).permute(0,3,1,2).float()
        inter = (pred_prob * target).sum(dim=(2,3))
        denom = pred_prob.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * inter + smooth) / (denom + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        if pred.size(1) == 1:
            ce = F.binary_cross_entropy_with_logits(pred, target.float())
            pred_prob = torch.sigmoid(pred)
            tgt_prob = target.float()
            cl = soft_clDice(pred_prob, tgt_prob)
        else:
            ce = self.ce(pred, target)
            # treat class 1 as foreground for clDice; adapt if needed
            pred_prob = torch.softmax(pred, dim=1)[:,1:2]
            tgt_prob = (target==1).unsqueeze(1).float()
            cl = soft_clDice(pred_prob, tgt_prob)
        dl = self.dice_loss(pred, target)
        return 0.6*ce + 0.4*dl + self.gamma*(1 - cl)

class CrackAugmentations:
    def __init__(self):
        pass

    def random_augmentation(self, image):
        augmentations = [
            self.blur,
            self.rotate,
            self.flip,
            self.random_crop,
            self.to_grayscale,
            self.to_rgb
        ]
        return augmentations[np.random.randint(0, len(augmentations))](image)


    def blur(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def rotate(self, image, angle=90):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((angle, angle)),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def flip(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def random_crop(self, image, crop_size = (500, 500)):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size),
            transforms.ToPILImage()
        ])
        return transform(image)
        
    def to_grayscale(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToPILImage()
        ])
        return transform(image)
    
    def to_rgb(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToPILImage()
        ])
        return transform(image)

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir = None, resize_size=(128, 128), sample_size=None, **kwargs):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        

        # Collect all images and masks (any jpg/png/jpeg)
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.[pjPJ][pnPN][gG]")))
        if masks_dir is not None:
            mask_files  = sorted(glob.glob(os.path.join(masks_dir,  "*.[pjPJ][pnPN][gG]")))

        # Build mapping base_name -> full path (for all extensions)
        image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        if masks_dir is not None:
            mask_map  = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

        # Keep only common bases
        if masks_dir is not None:
            common_bases = sorted(set(image_map.keys()) & set(mask_map.keys()))
        else:
            common_bases = sorted(set(image_map.keys()))

        self.img_paths = [image_map[b] for b in common_bases]
        if masks_dir is not None:
            self.mask_paths  = [mask_map[b] for b in common_bases]
        else:
            self.mask_paths = None

        if sample_size:
            self.img_paths = self.img_paths[:sample_size]
            if masks_dir is not None:
                self.mask_paths  = self.mask_paths[:sample_size]

        self.resize_size = resize_size
        # add any transforms or augmentation setup here

    def __len__(self):
        return len(self.img_paths)


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

        return img
    
if __name__ == "__main__":
    pass
    

class Pipeline:
    def __init__(self, images_path, ground_truths_path = None, resize_size=(256, 256), sample_size=None):
        self.images_path = images_path
        self.ground_truths_path = ground_truths_path
        self.resize_size = resize_size
        self.sample_size = sample_size

    def get_dataset(self):
        return CrackDataset(self.images_path, self.ground_truths_path, self.resize_size, self.sample_size, augmentation_percent=0.5)

    def load_model(self, model_path="models/unet.pth"):
        params = torch.load(model_path, map_location=torch.device('cpu'))
        model = UNet(n_classes=1)
        model.load_state_dict(params)
        return model
    
    def log_params(self, params):
        mlflow.log_params(params)

    def predict(self, model, dataset):
        transform = transforms.Compose([
            transforms.ToPILImage(),    
            transforms.Resize((533, 800)),
            transforms.ToTensor()
        ])
        model.eval()
        predictions, probabilities = [], []
        data = torch.utils.data.DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            for img in data:
                probs = model(img)
                probs = transform(probs.squeeze(0))
                output = probs.argmax(dim=0, keepdim=True)
                probabilities.append(probs)
                predictions.append(output)

        return predictions, probabilities
    
    def from_predictions_to_imgs(self, predictions):
        imgs = []
        for pred in predictions:
            if pred.ndim == 4:
                pred = pred.squeeze(0)
            pred_img = transforms.ToPILImage()(pred.float())
            imgs.append(pred_img)
        return imgs

    def add_n_classes(self, model, additional_classes):
        n_classes = model.n_classes + additional_classes
        if isinstance(model, UNet):
            model.n_classes = n_classes
            model.conv_last = nn.Conv2d(64, n_classes, 1)
        else:
            raise NotImplementedError("add_n_classes is not implemented for this model type.")
        return model
    
    def finetune(self, model, dataset: torch.utils.data.Dataset, epochs=50, additional_classes=None):
        if additional_classes is not None:
            model = self.add_n_classes(model, additional_classes)

        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        criterion = DiceCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Block all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last convolution block(s)
        for param in model.dconv_up1.parameters():
            param.requires_grad = True
        for param in model.conv_last.parameters():
            param.requires_grad = True

        # Training loop (single epoch for demonstration)
        pbar = tqdm.tqdm(range(epochs), desc="Finetuning Epochs")
        model.train()
        for epoch in pbar:
            loss_epoch = 0.0
            for images, masks in dataloader:
                images, masks = images.to("cpu"), masks.long().to("cpu")
                preds = model(images)
                loss = criterion(preds, masks.squeeze(1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.item()
            pbar.set_postfix({"loss": loss_epoch / len(dataloader)})
        return model
    
    def save_model(self, model, save_path="models/finetuned_unet.pth"):
        torch.save(model.state_dict(), save_path)
        return save_path
    
    def mask_to_polygons(self, mask, threshold=0.5):
        """
        Converts a binary mask (numpy array) to a list of polygons
        in normalized Label Studio format (0–100 scale).
        """
        # Binarize mask
        breakpoint()
        mask_bin = (mask > threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        h, w = mask.shape
        for cnt in contours:
            # Convert contour to normalized coordinates (Label Studio uses %)
            points = [[(x / w) * 100, (y / h) * 100] for [x, y] in cnt.squeeze().tolist()]
            polygons.append(points)
        return polygons
    
    

    def polygon_to_labelstudio(self, points, label="Crepe", image_w=1000, image_h=667):
        return {
            "id": str(uuid.uuid4())[:10],
            "type": "polygonlabels",
            "value": {
                "points": points,
                "polygonlabels": [label],
            },
            "to_name": "image",
            "from_name": "label",
            "origin": "model",
            "image_rotation": 0,
            "original_width": image_w,
            "original_height": image_h
        }

    
    def from_predictions_to_json(self, predictions):
        json_results = []
        for pred in predictions:
            mask = pred.squeeze(0).numpy()
            polygons = self.mask_to_polygons(mask)
            json_results.append(self.polygon_to_labelstudio(polygons))
        return json_results

if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]
    for n in nums:
        if n ==3:
            break
    else:
        print("Completed without break")
    model_path = "models/unet.pth"
    pipeline = Pipeline(images_path="data/A22/images/", ground_truths_path="data/A22/masks/", resize_size=(256, 256), sample_size=100)
    dataset = pipeline.get_dataset()
    model = pipeline.load_model(model_path=model_path)
    model = pipeline.finetune(model, dataset, epochs=50, additional_classes=2)
    pipeline.save_model(model)
    pipeline.ground_truths_path = None
    dataset = pipeline.get_dataset()
    predictions, probabilities = pipeline.predict(model, dataset)
   
    pred_imgs = pipeline.from_predictions_to_imgs(predictions)
    prob_imgs = pipeline.from_predictions_to_imgs(probabilities)
    
    for pred, prob in zip(pred_imgs, prob_imgs):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Predicted Mask")
        plt.imshow(pred)
        plt.subplot(1,2,2)
        plt.title("Predicted Probabilities")
        plt.imshow(prob)
        plt.show()
    