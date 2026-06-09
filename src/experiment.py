import mlflow.pytorch
import numpy as np
import cv2
import os
import mlflow
from networks import *
from probabilistic_circuits.probabilistic_circuits import *
from probabilistic_circuits.grammatical_evolution import *
from nnunet_pipeline import *
import abc
from torch.utils.data import DataLoader
import tqdm
from tqdm import trange
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import *
import pickle
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from skimage.filters import frangi

from joblib import Parallel, delayed
import json

import pandas as pd
from crackdata_processing import CrackDataset

from cv_modules.factories import CVModuleFactory

from probabilistic_circuits.grammatical_evolution_unsupervised import UnsupervisedGEOptimizer

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch import multiprocessing as mp
import multiprocessing
import time

def average_gradients_across_workers(params):
    """All-reduce gradients in-place and divide by world_size."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def run_experiment(config):
    if config["algorithm"] == "unet":
        experiment = UNetExperiment(config)
        experiment.run()
    
    elif config["algorithm"] == "unetplus":
        experiment = ExperimentUnetPlus(config)
        experiment.run()
    
    # elif config["algorithm"] == "nnunet":
    #     experiment = ExperimentNNUNet(config)
    #     experiment.run()

    elif config["algorithm"] == "convae":
        experiment = ConvAEExperiment(config)
        experiment.run()

    elif config["algorithm"] == "crackformer":
        experiment = ExperimentCrackFormer(config)
        experiment.run()

    elif config["algorithm"] == "deepcrackz":
        experiment = ExperimentDeepCrackZ(config)
        experiment.run()

    elif config["algorithm"] == "pcnet":
        experiment = ExperimentPC(config)
        experiment.run()

    elif config["algorithm"] == "pcnet_ge":
        experiment = ExperimentGEPC(config)
        experiment.run()

    elif config["algorithm"] == "pcnet_ge_unsupervised":
        experiment = ExperimentUnsupervisedGEPC(config)
        experiment.run()

    else:
        raise ValueError(f"Algorithm {config['algorithm']} not supported")
    
# ============================================================
# Loss Functions
# ============================================================

class DiceLoss(nn.Module):
    """Combination of CrossEntropy and Dice loss"""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=pos_weight)

    def dice_loss(self, pred, target, smooth=1.0):
        if pred.size(1) == 1:  # binary
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:  # multi-class
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=pred.size(1))
            target = target.permute(0, 3, 1, 2).float()

        intersection = (pred_prob * target).sum(dim=(2, 3))
        denominator = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (denominator + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        return  0.6* self.ce(pred, target) + 0.4*self.dice_loss(pred, target)

class LogDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def log_dice_loss(self, pred, target):
        C = pred.size(1)
        if C == 1:
            pred_prob = torch.sigmoid(pred)
            target = target.unsqueeze(1).float()
        else:
            pred_prob = torch.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (pred_prob * target).sum(dim=(2, 3))
        denominator = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        loss = -torch.log(torch.clamp(dice, min=1e-6))
        return loss.mean()

    def focal_loss(self, pred, target):
        """
        Generalized Focal Loss using torch.nn.functional.cross_entropy only.
        Works for both binary (C=1) and multi-class (C>1) segmentation.

        Args:
            pred:   [B, C, H, W] logits
            target: [B, H, W] or [B, 1, H, W] integer class labels
        """
        # If target has channel dimension, squeeze it
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()
        # Handle binary as 2-class cross-entropy (C=1 → expand to C=2)
        if pred.size(1) == 1:
            # Convert single-logit binary case to two-class logits for CE
            pred = torch.cat([pred, -pred], dim=1)

        # Compute per-pixel cross entropy (no reduction)
        ce_loss = F.cross_entropy(pred, target.long(), reduction="none")  # [B, H, W]

        # Compute pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Alpha handling
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        elif isinstance(self.alpha, torch.Tensor) and self.alpha.numel() == pred.size(1):
            alpha_t = self.alpha[target]  # [B, H, W]
        else:
            raise ValueError("alpha must be scalar or a per-class tensor with length = n_classes")

        # Focal loss
        focal = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


    def forward(self, pred, target):
        log_dice = self.log_dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return 0.5 * log_dice + 0.5 * focal
    
import torch
import torch.nn.functional as F

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
            if target.ndim == 3:
                target = target.unsqueeze(1)
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
        return 0.2*ce + 0.1*dl + self.gamma*(1 - cl)


class UnsupervisedPCNetLoss(nn.Module):
    """
    A stable, balanced, and segmentation-oriented unsupervised loss for crack detection.

    Key features:
      • symmetric background/crack distributions
      • gamma entropy term prevents collapse
      • sparsity term anchors crack ratio to realistic level
      • deviation-based crack modeling
      • logit temperature softens responsibilities
      • works with 1-channel or 2-channel PCNet outputs
    """

    def __init__(self,
                 target_crack_ratio=0.06,   # expected 3% pixels are crack (tune)
                 lambda_sparsity=0.2,       # weight for crack area prior
                 lambda_entropy=0.15,       # weight for gamma entropy
                 eps=1e-6,
                 temperature=2.0):          # softens gamma to avoid saturation
        super().__init__()
        self.target_r = target_crack_ratio
        self.lambda_s = lambda_sparsity
        self.lambda_e = lambda_entropy
        self.eps = eps
        self.T = temperature

    def forward(self, logits, x):
        """
        logits: (B,1,H,W) or (B,2,H,W)
        x:      (B,C,H,W)
        """

        # ---------------------------------------------------------
        # 1. RESPONSIBILITY γ
        # ---------------------------------------------------------
        if logits.shape[1] == 1:
            crack_logits = logits[:, 0]
            background_logits = torch.zeros_like(crack_logits)
        else:
            background_logits = logits[:, 0]
            crack_logits      = logits[:, 1]

        # Temperature-scaled logit difference to avoid saturation
        delta = (crack_logits - background_logits) / self.T
        delta = torch.clamp(delta, -8, 8)  # prevents extreme γ
        gamma = torch.sigmoid(delta)       # (B,H,W)

        gamma_full = gamma.unsqueeze(1).expand(-1, x.size(1), -1, -1)

        # ---------------------------------------------------------
        # 2. BACKGROUND MODEL (Gaussian-like)
        # ---------------------------------------------------------
        mu_b  = torch.mean(x, dim=[2,3], keepdim=True)
        std_b = torch.abs(x - mu_b) + self.eps   # local variance (balanced!)

        log_pb = -0.5 * (((x - mu_b) / std_b)**2 +
                         torch.log(2 * torch.pi * std_b**2))

        # ---------------------------------------------------------
        # 3. CRACK MODEL (deviation-based)
        # ---------------------------------------------------------
        # cracks are deviations from the background mean
        deviation = torch.abs(x - mu_b)

        # cracks get a moderate variance model centered around deviations
        std_c = deviation + self.eps
        mu_c  = mu_b

        log_pc = -0.5 * (((x - mu_c) / std_c)**2 +
                         torch.log(2 * torch.pi * std_c**2))

        # ---------------------------------------------------------
        # 4. MIXTURE PROBABILITY
        # ---------------------------------------------------------
        nll = -(gamma_full * log_pc + (1 - gamma_full) * log_pb)  # (B,C,H,W)
        nll = nll.mean(dim=1)                                     # (B,H,W)
        base_loss = nll.mean()

        # ---------------------------------------------------------
        # 5. SPARSITY PRIOR: penalize deviation from target crack ratio
        # ---------------------------------------------------------
        crack_ratio = gamma.mean()
        sparsity_prior = (crack_ratio - self.target_r).pow(2)

        # ---------------------------------------------------------
        # 6. ENTROPY REGULARIZATION on γ: keeps responsibilities soft
        # ---------------------------------------------------------
        entropy = -(gamma * torch.log(gamma + self.eps) +
                    (1 - gamma) * torch.log(1 - gamma + self.eps))
        entropy = entropy.mean()

        # ---------------------------------------------------------
        # FINAL LOSS
        # ---------------------------------------------------------
        loss = (
            base_loss +
            self.lambda_s * sparsity_prior -
            self.lambda_e * entropy          # **maximize entropy**
        )

        return loss



    
class Experiment:
    def __init__(self, config):
        self.config = config
        self.execution_mode = self.config.get("execution_mode", "train")
        self.device = self.config.get("device", "cpu")

        self.data_path = self.config.get("data_path", "data/train")
        self.validation_split = self.config.get("validation_split", 0.2)
        self.test_path = self.config.get("test_path", "data/test")
        self.save_path = self.config.get("save_path", "models/")

        self.batch_size = self.config.get("batch_size", 16)
        self.sample_size = self.config.get("sample_size", None)
        self.test_size = self.config.get("test_size", None)
        self.resize_size = self.config.get("resize_size", [256, 256])
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.threshold = self.config.get("threshold", 0.5)
        self.n_classes = self.config.get("n_classes", 2)
        self.preprocessing = self.config.get("preprocessing", False)
        
        self.epochs = self.config.get("epochs", 10)
        self.model = None
        self.lr = self.config.get("learning_rate", 0.001)

        self.log_path = self.config.get("log_dir", "logs/")
        print_configs("Experiment Configurations:", self.config)
        self.experiment_dict = {}
        self.experiment_report = os.path.join(self.log_path, "experiment_report.txt")
        with open(self.experiment_report, "w") as f:
            f.write(f"Experiment Configurations: {self.config}\n")
        f.close()
    

    
    @abc.abstractmethod
    def set_algorithm(self):
        pass

    def set_dataset(self,train_path, test_path=None):

        if self.execution_mode == "train":
            self.images_path = train_path + "/images/"
            self.ground_truth_path = train_path + "/masks/"  
            if self.sample_size is not None: 
                n_images = len(os.listdir(self.images_path))
                print_info(f"Number of images: {n_images}")
                self.sample_size = int(n_images * self.sample_size)
                print_info(f"Using sample size: {self.sample_size}/{n_images} which is {self.sample_size/n_images*100}% of the dataset")  
            self.data = CrackDataset(self.images_path, self.ground_truth_path,sample_size=self.sample_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
            torch.manual_seed(self.config.get("seed", 42))
            perm = torch.randperm(len(self.data))  # permutation over CrackDataset

            if self.execution_mode == "train":
                if self.validation_split > 0:
                    split_idx = int((1 - self.validation_split) * len(self.data))
                    train_idx = perm[:split_idx]
                    val_idx = perm[split_idx:]
                    self.train_data = torch.utils.data.Subset(self.data, train_idx)
                    self.val_data = torch.utils.data.Subset(self.data, val_idx)
                    print_info(f"Training samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}")
                else:
                    self.train_data = torch.utils.data.Subset(self.data, perm)
                    self.val_data = None
                    print_info(f"Training samples: {len(self.train_data)}")
            if test_path is not None:
                self.test_images_path = test_path + "/images/"
                self.test_ground_truth_path = test_path + "/masks/"
                self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
                print_info(f"Test samples: {len(self.test_data)}")
            self.experiment_dict["dataset_size"] = {
                "train": len(self.data),
                "val": len(self.val_data) if self.validation_split > 0 else 0,
                "test": len(self.test_data) if test_path is not None else 0
            }
        elif self.execution_mode == "test":
            self.images_path = test_path + "/images/"
            self.ground_truth_path = test_path + "/masks/"
            self.test_data = CrackDataset(self.test_images_path, self.test_ground_truth_path,sample_size=self.test_size, resize_size=(self.resize_size[0], self.resize_size[1]), preprocessing=self.preprocessing)
            print_info(f"Test samples: {len(self.test_data)}")
            self.experiment_dict["dataset_size"] = {
                "test": len(self.test_data)
            }

        else:
            raise ValueError(f"Execution mode {self.execution_mode} not supported")
        
        with open(self.experiment_report, "a") as f:
            f.write(f"Dataset sizes: {self.experiment_dict['dataset_size']}\n")
        f.close()

    # ----------------------------
    # Metrics
    # ----------------------------

    def dice_coef(self, logits, targets, smooth=1.0):
        """
        logits:  (B, C, H_pred, W_pred)
        targets: (B, H_gt,  W_gt) or (H, W)
        """

        # ---------------------------------------------------------
        # 1. Make sure targets has batch dimension
        # ---------------------------------------------------------
        if targets.dim() == 2:  # (H, W)
            targets = targets.unsqueeze(0)  # (1, H, W)

        # ---------------------------------------------------------
        # 2. Resize logits to match GT spatial size
        # ---------------------------------------------------------
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        B, C, H, W = logits.shape

        # ---------------------------------------------------------
        # 3. Binary Dice
        # ---------------------------------------------------------
        if C == 1:
            probs = torch.sigmoid(logits)              # (B,1,H,W)
            targets = targets.unsqueeze(1).float()     # (B,1,H,W)

            intersection = (probs * targets).sum(dim=(2,3))
            denominator   = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))

            dice = (2 * intersection + smooth) / (denominator + smooth)
            return dice.mean()

        # ---------------------------------------------------------
        # 4. Multi-class Dice (macro Dice)
        #    One-hot encode ground truth
        # ---------------------------------------------------------
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)

        # one-hot: (B,H,W,C) → (B,C,H,W)
        targets_1h = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_1h).sum(dim=(2,3))      # (B,C)
        denominator  = probs.sum(dim=(2,3)) + targets_1h.sum(dim=(2,3))

        dice = (2 * intersection + smooth) / (denominator + smooth)  # (B,C)

        # Mean across classes + batch
        return dice.mean()




    def iou_score(self, logits, targets, num_classes=None):
        """
        logits:  (N, C, H_pred, W_pred)
        targets: (N, H_gt,  W_gt)
        """

        # ----------------------------------------------
        # 1. Resize logits if spatial dimensions differ
        # ----------------------------------------------
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # ----------------------------------------------
        # 2. Convert logits → predicted class indices
        # ----------------------------------------------
        preds = torch.argmax(logits, dim=1)   # (N, H, W)

        # ----------------------------------------------
        # 3. Flatten for vectorized IoU computation
        # ----------------------------------------------
        preds = preds.view(-1)
        targets = targets.view(-1)

        # ----------------------------------------------
        # 4. Determine number of classes
        # ----------------------------------------------
        if num_classes is None:
            num_classes = int(logits.shape[1])

        # ----------------------------------------------
        # 5. Compute confusion matrix
        # ----------------------------------------------
        confusion = torch.bincount(
            num_classes * targets + preds,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        TP = confusion.diag()
        FP = confusion.sum(0) - TP
        FN = confusion.sum(1) - TP

        IoU = TP / (TP + FP + FN + 1e-7)
        mIoU = IoU.mean()
        return mIoU



    def cl_iou_score(self,pred_mask, true_mask, tau=4):
        """
        pred_mask, true_mask: (B,1,H,W)
        """
        B = pred_mask.shape[0]
        device = pred_mask.device
        pred_mask = (pred_mask > 0.5).float()
        true_mask = (true_mask > 0.5).float()

        # Skeletonization approximation
        kernel_3 = torch.ones((1, 1, 3, 3), device=device)
        eroded_true = F.max_pool2d(1 - true_mask, 3, 1, 1)
        eroded_pred = F.max_pool2d(1 - pred_mask, 3, 1, 1)
        skel_true = torch.relu(true_mask - (1 - eroded_true))
        skel_pred = torch.relu(pred_mask - (1 - eroded_pred))

        # --- Dilation ---
        k_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tau+1, 2*tau+1))
        k = torch.tensor(k_np, dtype=torch.float32, device=device)[None, None, :, :]
        dil_true = (F.conv2d(skel_true, k, padding=tau, groups=1) > 0).float()
        dil_pred = (F.conv2d(skel_pred, k, padding=tau, groups=1) > 0).float()

        TPcl = (skel_true * dil_pred).sum(dim=(1,2,3))
        FPcl = (skel_pred * (1 - dil_true)).sum(dim=(1,2,3))
        FNcl = (skel_true * (1 - dil_pred)).sum(dim=(1,2,3))

        cl_iou = TPcl / (TPcl + FPcl + FNcl + 1e-8)
        return cl_iou.mean().item()


    from torch.utils.data import DataLoader, TensorDataset

    def get_dataloaders(self, data, batch_size=None, labels=None, train_sampler=None, shuffle=True, num_workers=0):
        """
        If train_sampler is provided, shuffle must be False (sampler handles shuffling).
        """
        if labels is not None:
            dataset = TensorDataset(data, labels)
        else:
            dataset = data

        if train_sampler is not None:
            return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    
    def compute_pos_weight(self, masks):
        """
        Computes pos_weight = num_neg / num_pos for BCEWithLogitsLoss
        from a segmentation dataset.
        """
        # Class balance
        flat = masks.view(-1)
        counts = torch.bincount(flat, minlength=self.n_classes).float()
        counts[counts == 0] = 1.0
        balance = (1.0 / counts)
        balance = balance / balance.sum()
        return balance

    @abc.abstractmethod
    def setup(self, pos_weight=1.0):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def plot_losses(self, train_losses, val_losses=None):
        plt.plot(train_losses, label="Train Loss")
        if val_losses:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.show()

    def log_results(self):
        pass




class ExperimentDL(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, pos_weight=1.0):
        self.set_optimizer(torch.optim.Adam(self.model.parameters(), lr=self.learning_rate))
        self.set_scheduler(torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1))
        self.set_loss(DiceCELoss())

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_loss(self, loss):
        self.loss = loss
    
    
    def get_model_size_kb(self,model: torch.nn.Module):
        """Return model size in kilobytes (KB)."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_bytes = param_size + buffer_size
        return total_size_bytes / 1024  # Convert to KB
    # ----------------------------
    # Training loop
    # ----------------------------

    def iou_score(self, logits, targets, num_classes=None):

        # Resize if sizes do not match
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # Decide number of classes
        if num_classes is None:
            num_classes = 2 if logits.shape[1] == 1 else logits.shape[1]

        # Predictions
        if logits.shape[1] == 1:   # binary
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        else:                      # multiclass
            preds = torch.argmax(logits, dim=1)

        # Flatten + convert
        preds = preds.view(-1).long()
        targets = targets.view(-1).long()

        # Compute confusion matrix
        bins = num_classes * targets + preds
        confusion = torch.bincount(
            bins,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        TP = confusion.diag()
        FP = confusion.sum(0) - TP
        FN = confusion.sum(1) - TP

        IoU = TP / (TP + FP + FN + 1e-7)
        return IoU.mean()

    def train(self, model, train_data, val_data=None, 
            loss_function=None, optimizer=None, scheduler=None, 
            device="cuda", epochs=20, batch_size=8):

        # Default loss: BCE + Dice
        if loss_function is None:
            loss_function = DiceCELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = self.get_dataloaders(train_data, batch_size=batch_size)
        val_loader = self.get_dataloaders(val_data, batch_size=batch_size) if val_data else None

        model.to(device)
        train_losses, val_losses, train_dices, train_clious, val_dices, train_ious, val_ious, val_clious = [], [], [], [], [], [], [] , []
        
        for epoch in trange(epochs):
            model.train()
            epoch_loss, epoch_dice, epoch_iou, epoch_cl_iou = 0, 0, 0, 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks
                optimizer.zero_grad()
                outputs = model(images)  
                loss = loss_function(outputs, masks) 
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_dice += self.dice_coef(outputs, masks).item()
                epoch_iou  += self.iou_score(outputs, masks).item()
                epoch_cl_iou += self.cl_iou_score(outputs, masks)

            train_loss = epoch_loss / len(train_loader)
            train_dice = epoch_dice / len(train_loader)
            train_iou  = epoch_iou / len(train_loader)
            train_cl_iou = epoch_cl_iou / len(train_loader)
            train_losses.append(train_loss)
            train_dices.append(train_dice)
            train_ious.append(train_iou)
            train_clious.append(train_cl_iou)
            # Validation
            if val_loader:
                model.eval()
                val_loss, val_dice, val_iou, val_cl_iou = 0, 0, 0, 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        val_loss += loss_function(outputs, masks).item()
                        val_dice += self.dice_coef(outputs, masks).item()
                        val_iou  += self.iou_score(outputs, masks).item()
                        val_cl_iou += self.cl_iou_score(outputs, masks)

                val_loss /= len(val_loader)
                val_dice /= len(val_loader)
                val_iou  /= len(val_loader)
                val_cl_iou /= len(val_loader)

            else:
                val_loss, val_dice, val_iou = None, None, None
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)
            val_clious.append(val_cl_iou)
            if scheduler:
                scheduler.step()

            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, CL IoU: {train_cl_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}, CL IoU: {val_cl_iou:.4f}" if val_loader else "")
            
        train_df = pd.DataFrame({"train_loss": train_losses, "train_dice": train_dices, "train_iou": train_ious, "train_cl_iou": train_clious})
        train_df.to_csv(os.path.join(self.log_path, "training.csv"), index=False)
        if val_loader:
            val_df = pd.DataFrame({"val_loss": val_losses, "val_dice": val_dices, "val_iou": val_ious, "val_cl_iou": val_clious})
            val_df.to_csv(os.path.join(self.log_path, "validation.csv"), index=False)
        
        
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training Loss: {train_losses[-1]}, Dice: {train_dices[-1]}, IoU: {train_ious[-1]}, CL IoU: {train_clious[-1]}\n")
            if val_loader:
                f.write(f"Final Validation Loss: {val_losses[-1]}, Dice: {val_dices[-1]}, IoU: {val_ious[-1]}, CL IoU: {val_clious[-1]}\n")
        f.close()
        return model

    def finetune(self, model, dataloader, n_classes=2):
        model.load_state_dict(torch.load("unet_pretrained.pth"))

        # Replace the last layer for new classes
        model.conv_last = nn.Conv2d(64, n_classes, 1)

        # Option 1: freeze earlier layers (useful if the dataset is small)
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last convolution block(s)
        for param in model.dconv_up1.parameters():
            param.requires_grad = True
        for param in model.conv_last.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        criterion = self.loss

        for images, masks in dataloader:
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def validate(self, model, val_loader, loss_function):
        model.eval()
        epoch_val_loss = 0  
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                epoch_val_loss += loss.item()
        model.train()
        return epoch_val_loss

    def evaluate(self, model, test_data, loss_function=torch.nn.BCEWithLogitsLoss()):
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        model.eval()
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
        
                dice_score = self.dice_coef(outputs, masks)
                iou_score = self.iou_score(outputs, masks)
                cl_iou_score = self.cl_iou_score(outputs, masks)
                prob = torch.sigmoid(outputs)
                pred_mask = (prob > self.threshold).float()
                true_mask = masks.float()

                # Flatten predictions and targets
                pm = pred_mask.view(-1).long()
                tm = true_mask.view(-1).long()

                tp = ((pm == 1) & (tm == 1)).sum()
                fp = ((pm == 1) & (tm == 0)).sum()
                fn = ((pm == 0) & (tm == 1)).sum()
                tn = ((pm == 0) & (tm == 0)).sum()

                # Accuracy
                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp + 1e-8)

                # Recall = TP / (TP + FN)
                recall = tp / (tp + fn + 1e-8)

                # F1 = harmonic mean
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                losses.append(loss.item())
                dices.append(float(torch.as_tensor(dice_score).mean()))
                ious.append(float(torch.as_tensor(iou_score).mean()))
                clious.append(float(torch.as_tensor(cl_iou_score)))

                accuracies.append(accuracy.item())
                precisions.append(precision.item())
                recalls.append(recall.item())
                f1_scores.append(f1_score.item())

        print_info("✅ Testing complete.")
        print_info(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}")
        print_info(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}")
        self.experiment_dict["results"] = {
            "test_loss": np.mean(losses),
            "test_dice": np.mean(dices),
            "test_iou": np.mean(ious),
            "test_cl_iou": np.mean(clious),
            "test_accuracy": np.mean(accuracies),
            "test_precision": np.mean(precisions),
            "test_recall": np.mean(recalls),
            "test_f1_score": np.mean(f1_scores)
        }
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.experiment_dict["n_trainable_params"] = n_trainable_params
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}\n")
            f.write(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}\n")
        test_loss = sum(losses)

                
        return test_loss/len(test_loader)        

    def log_results(self):
        pass

    def show_predictions(self, model, dataset, device, num_samples=3, threshold=0.5):
        # Save model info
        model_json = os.path.join(self.log_path, "model_info.json")
        model_memory = self.get_model_size_kb(model)
        with open(self.experiment_report, "a") as f:
            f.write(f"Model info: {model_json}\n")
            f.write(f"Model size: {model_memory} KB\n")
        f.close()

        
        predictions_dir = os.path.join(self.log_path, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # We’ll create up to num_samples * 3 total images per grid
        total_samples = len(dataset)
        num_groups = total_samples // num_samples
        max_pred = 15
        for j in range(num_groups):
            if j > max_pred:
                print(f"Saved {max_pred} prediction grids.")
                break
            fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 9))  # 3 rows × num_samples columns

            for i in range(num_samples):
                idx = j * num_samples + i
                if idx >= total_samples:
                    # Hide unused axes if dataset doesn't divide evenly
                    for row in range(3):
                        axes[row, i].axis("off")
                    continue

                img, gt = dataset[idx]
                img = img.unsqueeze(0).to(device)
                gt_np = gt.squeeze().cpu().numpy()

                # Prediction
                with torch.no_grad():
                    pred = model(img).cpu().squeeze()
                    prob = torch.sigmoid(pred).numpy()
                pred_mask = (prob > threshold).astype("uint8")

                # Convert input image for plotting
                img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

                # Row 1: Input
                axes[0, i].imshow(img_np)
                axes[0, i].set_title(f"Input {idx}")
                axes[0, i].axis("off")

                # Row 2: Ground Truth
                axes[1, i].imshow(gt_np, cmap="gray")
                axes[1, i].set_title(f"Ground Truth {idx}")
                axes[1, i].axis("off")

                # Row 3: Prediction
                axes[2, i].imshow(pred_mask, cmap="cool")
                axes[2, i].set_title(f"Prediction {idx}")
                axes[2, i].axis("off")
            plt.savefig(os.path.join(predictions_dir, f"predictions_grid_{j}.png"), bbox_inches="tight")
            plt.close(fig)
            

    @abc.abstractmethod
    def save_model(self, model_name="model"):
        pass

    @abc.abstractmethod
    def load_model(self, model_name="model"):
        pass

    def run(self):
        self.set_algorithm()
        self.model.to(self.device)
        
        
        if self.execution_mode == "train":
            self.set_dataset(self.data_path, self.test_path)
            # get the class imbalance ratio
            images = [img for img, mask in self.train_data]
            masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            masks = torch.from_numpy(masks).long().to(self.device)
            pos_weight = self.compute_pos_weight(masks)
            pos_weight = pos_weight[1]/pos_weight[0]
            self.setup(pos_weight=pos_weight)
            self.model = self.train(self.model, train_data=self.train_data, val_data=self.val_data, loss_function=self.loss, optimizer=self.optimizer, device=self.device, epochs=self.epochs, batch_size=self.batch_size)
            self.save_model(self.model_name)
            if self.test_path is not None:
                self.evaluate(self.model, self.test_data, loss_function=self.loss)
                json.dump(self.experiment_dict, open(os.path.join(self.log_path, "experiment_results.json"), "w"), indent=4)
                self.show_predictions(self.model, self.test_data, self.device, num_samples=2, threshold=self.threshold)
        elif self.execution_mode == "finetune":
            self.set_dataset(self.data_path, self.test_path)
            pos_weight = self.compute_pos_weight(self.data)
            self.set_algorithm()
            self.load_model()
            self.setup(pos_weight=pos_weight)
            self.model.to(self.device)
            self.finetune()

        elif self.execution_mode == "test":
            self.set_dataset(self.data_path, self.test_path)
            pos_weight = self.compute_pos_weight(self.data)
            self.set_algorithm()
            self.load_model(self.model_name)
            self.setup(pos_weight=pos_weight)
            self.model.to(self.device)
            print(self.model.info())
            self.evaluate(self.model, self.test_data, loss_function=self.loss)
            json.dump(self.experiment_dict, open(os.path.join(self.log_path, "experiment_results.json"), "w"), indent=4)
            self.show_predictions(self.model, self.test_data, self.device, num_samples=2, threshold=self.threshold)

class UNetExperiment(ExperimentDL):

    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")

    def set_algorithm(self):
        self.model = UNet(n_classes=1)
    
    def save_model(self, model_name="unet"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="unet"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ConvAEExperiment(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.input_channels = self.config.get("input_channels", 3)
        self.hidden_channels = self.config.get("hidden_channels", (128, 64))
        self.n_classes = self.config.get("n_classes", 1)
        self.kernel_size = self.config.get("kernel_size", 3)
        self.stride = self.config.get("stride", 1)
        self.padding = self.config.get("padding", 9)
        self.dropout = self.config.get("dropout", 0.2)
        self.model_name = self.config.get("model_name", "convae")
        self.filter_size = self.config.get("filter_size", (3,7))

    
    def set_algorithm(self):
        self.model = ConvAutoEncoder(input_channels=self.input_channels, hidden_channels=self.hidden_channels, n_classes=self.n_classes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dropout=self.dropout)

    def save_model(self, model_name="convae"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="convae"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentUnetPlus(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unetplus")
        

    def set_algorithm(self):
        self.model = UNetPlusPlus(n_classes=1)

    def save_model(self, model_name="unetplus"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="unetplus"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentCrackFormer(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "crackformer")
    
    def set_algorithm(self):
        self.model = crackformer()

    def save_model(self, model_name="crackformer"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="crackformer"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

class ExperimentDeepCrackZ(ExperimentDL):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "deepcrackL")
    
    def set_algorithm(self):
        self.model = DeepCrack()

    def save_model(self, model_name="deepcrackL"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{model_name}.pth"))

    def load_model(self, model_name="deepcrackL"):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{model_name}.pth")))

# import os
# import json
# import torch
# from torch.utils.data import Subset

# from crackdata_processing import CrackDataset  # your class
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# class nnUNetTrainer_Epochs(nnUNetTrainer):
#     """
#     Custom trainer to control 'epochs' in nnUNet v2 by defining:
#       - num_iterations_per_epoch
#       - num_epochs
#     Total training iterations = num_epochs * num_iterations_per_epoch
#     """

#     def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
#                  device=None, **kwargs):
#         super().__init__(plans, configuration, fold, dataset_json,
#                          unpack_dataset=unpack_dataset, device=device, **kwargs)

#         # Read custom values passed via kwargs (nnUNet forwards unknown args into trainer kwargs)
#         self._custom_num_epochs = int(kwargs.get("epochs", 100))
#         self._custom_iters_per_epoch = int(kwargs.get("num_iterations_per_epoch", 250))

#     @property
#     def num_iterations_per_epoch(self) -> int:
#         return self._custom_iters_per_epoch

#     @property
#     def num_epochs(self) -> int:
#         return self._custom_num_epochs


# class ExperimentNNUNet(Experiment):
#     def __init__(self, config):
#         super().__init__(config)

#         # nnUNet config
#         self.dataset_id = int(config.get("nnunet_dataset_id", 777))
#         self.dataset_name = config.get("nnunet_dataset_name", "Cracks")
#         self.nn_cfg = config.get("nnunet_config", "2d")
#         self.folds = config.get("nnunet_folds", [0])
#         self.num_gpus = int(config.get("nnunet_num_gpus", 1))
#         self.trainer = config.get("nnunet_trainer", "nnUNetTrainer")
#         self.plans = config.get("nnunet_plans", "nnUNetPlans")

#         # Ensure writable nnUNet dirs (falls back to ./nnunet_data/* if needed)
#         resolve_nnunet_paths(self.config)

#         self.nnUNet_raw = self.config["nnUNet_raw"]
#         self.nnUNet_preprocessed = self.config["nnUNet_preprocessed"]
#         self.nnUNet_results = self.config["nnUNet_results"]

#         self._env = os.environ.copy()
#         self._env["nnUNet_raw"] = self.nnUNet_raw
#         self._env["nnUNet_preprocessed"] = self.nnUNet_preprocessed
#         self._env["nnUNet_results"] = self.nnUNet_results


#         self.model_name = config.get("model_name", "nnunet")  # only for logging labeling

#     def set_algorithm(self):
#         # nnUNet decides architecture/training internally
#         pass

#     def setup(self, pos_weight=1.0):
#         pass

#     def save(self):
#         pass

#     def load(self):
#         pass

#     def validate(self):
#         pass

#     def resolve_sample_count(self, sample_size, n_total: int) -> int:
#         if sample_size is None:
#             return n_total
#         s = float(sample_size)
#         if 0 < s <= 1:
#             return max(1, int(round(s * n_total)))
#         return int(min(s, n_total))

#     def _build_train_subset(self):
#         full = self.data
#         n = len(full)
#         k = self.resolve_sample_count(self.sample_size, n)

#         g = torch.Generator()
#         g.manual_seed(int(self.config.get("seed", 42)))
#         idx = torch.randperm(n, generator=g)[:k].tolist()
#         return torch.utils.data.Subset(full, idx)

#     def set_dataset(self, train_path, test_path=None):
#         # Keep your same folder conventions

#         self.images_path = os.path.join(train_path, "images")
#         self.ground_truth_path = os.path.join(train_path, "masks")

#         self.data = CrackDataset(
#             self.images_path,
#             self.ground_truth_path,
#             sample_size=None,  # we handle sample_size via Subset for seed control
#             resize_size=(self.resize_size[0], self.resize_size[1]),
#             preprocessing=self.preprocessing
#         )

#         self.train_data = self._build_train_subset()

#         # nnUNet doesn't need your val split; it does its own CV folds.
#         # We keep val_data for compatibility, but don't use it.
#         self.val_data = None

#         if test_path is not None:
#             test_images = os.path.join(test_path, "images")
#             test_masks = os.path.join(test_path, "masks")
#             self.test_data = CrackDataset(
#                 test_images,
#                 test_masks,
#                 sample_size=self.test_size,
#                 resize_size=(self.resize_size[0], self.resize_size[1]),
#                 preprocessing=self.preprocessing
#             )
#         else:
#             self.test_data = None

#     def train(self):
#         # export subset to nnUNet_raw
#         export_crackdataset_subset_to_nnunet_raw(
#             train_ds=self.train_data,
#             test_ds=self.test_data,
#             nnunet_raw_dir=self.nnUNet_raw,
#             dataset_id=self.dataset_id,
#             dataset_name=self.dataset_name,
#             input_channels=int(self.config.get("input_channels", 3))
#         )

#         nnunet_plan_preprocess(self.dataset_id, env=self._env)

#         # You can train "all" folds for proper nnUNet CV, but it’s expensive.
#         # If you want faster learning-curve sweeps, train only fold 0:
#         folds_to_train = self.config.get("nnunet_train_folds", "0")  # "all" or "0"
#         device = self.config.get("nnunet_device", None)  # allow override
#         nnunet_train(
#             dataset_id=self.dataset_id,
#             cfg=self.nn_cfg,
#             folds=folds_to_train,
#             trainer=self.trainer,
#             plans=self.plans,
#             num_gpus=self.num_gpus,
#             env=self._env,
#             device=device,
#             num_epochs=int(self.config.get("epochs",80)),
#             num_iterations_per_epoch=int(self.config.get("nnunet_iter_per_epoch", 250)),
#         )

#     def evaluate(self):
#         if self.test_data is None:
#             print("[INFO] No test set provided; skipping.")
#             return None

#         ds_folder = f"Dataset{self.dataset_id:03d}_{self.dataset_name}"
#         raw_base = os.path.join(self.nnUNet_raw, ds_folder)
#         in_dir = os.path.join(raw_base, "imagesTs")

#         out_dir = os.path.join(self.nnUNet_results, ds_folder, "pred_imagesTs_seed" + str(self.config.get("seed", 0)))
#         nnunet_predict(
#             dataset_id=self.dataset_id,
#             in_dir=in_dir,
#             out_dir=out_dir,
#             cfg=self.nn_cfg,
#             folds=self.folds,
#             trainer=self.trainer,
#             plans=self.plans,
#             env=self._env
#         )

#         # compute Dice/IoU
#         dices, ious = [], []
#         for j in range(len(self.test_data)):
#             case_id = f"case_{100000 + j:06d}"
#             pred_path = os.path.join(out_dir, f"{case_id}.nii.gz")
#             pred = np.squeeze(nib.load(pred_path).get_fdata())  # (H,W) or float
#             pred01 = (pred > 0.5).astype(np.uint8)

#             _, gt = self.test_data[j]
#             if isinstance(gt, torch.Tensor):
#                 gt = gt.detach().cpu().numpy()
#             gt = np.squeeze(gt)
#             gt01 = (gt > 0.5).astype(np.uint8)

#             d, i = dice_iou_binary(pred01, gt01)
#             dices.append(d)
#             ious.append(i)

#         results = {
#             "test_dice": float(np.mean(dices)),
#             "test_iou": float(np.mean(ious)),
#             "n_train": int(len(self.train_data)),
#             "seed": int(self.config.get("seed", 0))
#         }

#         print(f"[nnUNet] n_train={results['n_train']} seed={results['seed']} "
#               f"Dice={results['test_dice']:.4f} IoU={results['test_iou']:.4f}")

#         # write to your log_dir
#         with open(os.path.join(self.log_path, "nnunet_results.json"), "w") as f:
#             json.dump(results, f, indent=2)

#         return results

#     def run(self):
        self.set_dataset(self.data_path, self.test_path)

        if self.execution_mode == "train":
            self.train()
            self.evaluate()

        elif self.execution_mode == "test":
            self.evaluate()

        else:
            raise ValueError(f"Execution mode {self.execution_mode} not supported for nnUNet")

class ExperimentPC(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "unet")
        self.input_channels = self.config.get("input_channels", 3)
        self.max_depth = self.config.get("max_depth", 5)
        self.max_branching = self.config.get("max_branching", 2)
        self.n_classes = self.config.get("n_classes", 2)
        self.input_nodes = []

        self.epochs = self.config.get("epochs", 10)
        self.cv_epochs = self.config.get("cv_epochs", 5)

        self.seed = self.config.get("seed", 42)
        self.feature_extractor = self.config.get("feature_extractor", "handcrafted")
        self.filter_sizes = self.config.get("filter_sizes", (3, 7, 15))
        self.n_cpus = self.config.get("n_cpus", None)
        self.batch_size = self.config.get("batch_size", 4)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.n_cv_modules = self.config.get("n_cv_modules", 32)
        self.n_filters = self.config.get("n_filters", 4)
        self.set_algorithm()
        self._setup_cpu_env()
        self.cv_factory = None


        # cache for conv filter bank tensor
        self.cached_filter_tensor = None

    # --------------------------------------------------------
    # Algorithm
    # --------------------------------------------------------
    def set_algorithm(self):
        self.model = PCNet(
            n_classes=self.n_classes,
            distribution=None,
            device="cpu",
            max_depth=self.max_depth,
            max_branching=self.max_branching,
            seed=self.seed,
            cv_module = None
        )

    # --------------------------------------------------------
    # Threading & environment setup
    # --------------------------------------------------------
    def _setup_cpu_env(self):
        if self.n_cpus is None or self.n_cpus < 1:
            self.n_cpus = multiprocessing.cpu_count()
        # unify threads across libs
        torch.set_num_threads(self.n_cpus)
        os.environ["OMP_NUM_THREADS"] = str(self.n_cpus)
        os.environ["MKL_NUM_THREADS"] = str(self.n_cpus)
        print(f"⚙️ Using {self.n_cpus} CPU cores (torch threads)")

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------
    def get_filter_bank(self):
        """Comprehensive crack-oriented filter bank."""
        filters = {
            # Basic edge filters
            "sobel_x": np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], np.float32),
            "sobel_y": np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], np.float32),
            "laplacian": np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], np.float32),
            "sharpen": np.array([[ 0, -1,  0],
                                 [-1,  5, -1],
                                 [ 0, -1,  0]], np.float32),
            "gauss3": (cv2.getGaussianKernel(3, 0.8) @
                       cv2.getGaussianKernel(3, 0.8).T).astype(np.float32),
        }

        # Directional Gabor filters
        for theta in np.linspace(0, np.pi, 6, endpoint=False):
            gabor = cv2.getGaborKernel((9, 9), 3.0, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
            filters[f"gabor_{int(theta * 180 / np.pi)}"] = gabor

        return filters

    # --------------------------------------------------------
    # Conv filter tensor building and caching
    # --------------------------------------------------------
    def _build_filter_tensor_once(self, filters, device="cpu"):
        if self.cached_filter_tensor is not None:
            return self.cached_filter_tensor

        filt_list = list(filters.values())
        max_h = max(k.shape[0] for k in filt_list)
        max_w = max(k.shape[1] for k in filt_list)

        def pad_to(k, H, W):
            out = np.zeros((H, W), np.float32)
            h, w = k.shape
            y0, x0 = (H - h) // 2, (W - w) // 2
            out[y0:y0 + h, x0:x0 + w] = k
            return out

        filt_padded = [pad_to(k, max_h, max_w) for k in filt_list]
        filt_tensor = torch.from_numpy(np.stack(filt_padded)).float().unsqueeze(1)
        filt_tensor = filt_tensor.repeat(1, 3, 1, 1).to(device)
        self.cached_filter_tensor = filt_tensor
        return self.cached_filter_tensor

    # --------------------------------------------------------
    # Convolutional feature computation
    # --------------------------------------------------------
    def compute_conv_features(self, images, filters, device="cpu"):
        if isinstance(images, list):
            images = torch.from_numpy(np.stack(images)).float()
        elif isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Normalize to BCHW (3 channels)
        if images.ndim != 4:
            raise ValueError(f"Expected 4D images, got {images.shape}")
        if images.shape[1] != 3 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] not in (1, 3):
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected image shape: {tuple(images.shape)}")

        images = images.to(device)
        filt_tensor = self._build_filter_tensor_once(filters, device=device)
        conv = F.conv2d(images, filt_tensor, padding="same")  # (B, K, H, W)
        return conv

    # --------------------------------------------------------
    # Multi-scale pooling
    # --------------------------------------------------------
    def multi_scale_pooling(self, feat_maps, pool_sizes=(3, 7, 15), modes=("avg", "max")):
        pooled = []
        for k in pool_sizes:
            if k <= 1:
                continue
            for mode in modes:
                if mode == "avg":
                    out = F.avg_pool2d(feat_maps, kernel_size=k, stride=1, padding=k // 2)
                elif mode == "max":
                    out = F.max_pool2d(feat_maps, kernel_size=k, stride=1, padding=k // 2)
                else:
                    raise ValueError(f"Unknown pooling mode: {mode}")
                pooled.append(out)
        return torch.cat(pooled, dim=1) if pooled else feat_maps

    # --------------------------------------------------------
    # Handcrafted features
    # --------------------------------------------------------
    def get_handcrafted_features(self, images, n_jobs=8, prune_features_idx=None):
        """Compute handcrafted features for images in parallel."""
        def process_one(img):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0

            # Normalize to HWC
            if img.ndim == 2:
                img_hwc = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                img_hwc = np.transpose(img, (1, 2, 0))
            elif img.ndim == 3 and img.shape[-1] == 3:
                img_hwc = img
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            gray = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                                cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            lab = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                               cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor((img_hwc * 255).astype(np.uint8),
                               cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

            feats = [
                gray,
                lab[..., 0], lab[..., 1], lab[..., 2],
                hsv[..., 0], hsv[..., 1], hsv[..., 2],
                cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1),
                cv2.Laplacian(gray, cv2.CV_32F),
            ]

            # Gabor filters (4 orientations)
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kernel = cv2.getGaborKernel((9, 9), 2.0, theta, 4.0, 0.5, 0, ktype=cv2.CV_32F)
                feats.append(cv2.filter2D(gray, cv2.CV_32F, kernel))

            # Optional Frangi (line enhancement)
            # if self.use_frangi:
            frangi_resp = frangi(gray, sigmas=(1, 5), scale_step=2,
                                    beta=0.5)
            feats.append(frangi_resp.astype(np.float32))

            # Optional High-pass
            # if self.use_highpass:
            blurred = cv2.GaussianBlur(gray, (0, 0), 5)
            highpass = cv2.subtract(gray, blurred)
            feats.append(highpass.astype(np.float32))

            return np.stack(feats, axis=0).astype(np.float32)

        workers = min(n_jobs if n_jobs else multiprocessing.cpu_count(), 16)
        print(f"⚙️ Using {workers} CPU workers for handcrafted feature extraction...")
        feats_list = Parallel(n_jobs=workers, backend="loky", verbose=10)(
            delayed(process_one)(img) for img in images
        )
        feats = np.stack(feats_list, axis=0)  # (B, C, H, W)

        # Conv bank (batched)
        conv_feats = self.compute_conv_features(images, self.get_filter_bank(), device=self.device)
        conv_feats = conv_feats.detach().cpu().numpy()

        base_feats = np.concatenate([feats, conv_feats], axis=1)
        base_tensor = torch.from_numpy(base_feats).float().to(self.device)

        pooled_feats = self.multi_scale_pooling(base_tensor,
                                                pool_sizes=self.filter_sizes,
                                                modes=("avg", "max"))
        pooled_feats = pooled_feats.cpu().numpy()
        final_feats = np.concatenate([base_feats, pooled_feats], axis=1)
        if prune_features_idx is not None:
            final_feats = final_feats[:, prune_features_idx, :, :]

        return final_feats

    # TODO one process per module
    def eval_cv_pop(self, modules, images):
        with multiprocessing.Pool(processes=self.n_cpus) as pool:
            results = pool.starmap(
                self._worker_process_wrapper,
                [(idx, module, images) for idx, module in enumerate(modules)]
            )

        return [fit for fit, _ in results]

    def _worker_process_wrapper(self, idx, module, imgs):
        
        print_debugging(f"[{time.strftime('%H:%M:%S')}] Worker {idx} starting...")
        try:
            feats = [module.get_output(img, return_map=True) for img in imgs]
            var = np.stack(feats).var()
            return (float(var), idx)
        except Exception as e:
            print(f"[Worker {idx}] error: {e}")
            return (float("-inf"), idx)

    def convolution_modules_tensor(
        self, images, cv_factory, return_maps=True, n_jobs=None, verbose=0, prune_features=None
    ):
        """
        Torchrun-friendly version with multiprocessing image parallelism.
        - Each rank processes a disjoint shard of images.
        - Each shard is processed in parallel using multiprocessing (per-image chunks).
        - Results are gathered with torch.distributed all_gather on tensors.
        - Output shape: [N_images, C, H, W] or [N_images, H, W] depending on your modules.
        """
        import multiprocessing as mp
        from joblib import Parallel, delayed
        import torch.distributed as dist
        import numpy as np
        import os

        # ---------------------------------------------------------
        # Normalize input to numpy
        # ---------------------------------------------------------
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = np.expand_dims(images, 0)  # [1, H, W]
        total_images = len(images)
        

        # ---------------------------------------------------------
        # Optimize module via variance (if no pruning provided)
        # ---------------------------------------------------------
        if prune_features is None:
            # ---------------------------------------------------------
            # CMA-ES optimization loop
            # ---------------------------------------------------------
            # Multiprocessing parallelization over MODULES
            modules = cv_factory.ask_pop()
            n_procs = min(n_jobs, len(modules))
            print_info(f"⚙️ Using {n_procs} processes for convolution module evaluation...")
            for epoch in trange(self.cv_epochs, desc="CV Module Optimization", disable=verbose == 0):
                print_debugging(f"Epoch {epoch}")
                modules = cv_factory.ask_pop()
                if not isinstance(modules, list):
                    modules = [modules]               

                fitness = self.eval_cv_pop(modules, images)

                print_debugging(f"Module fitness values: {fitness}")
                # CMA-ES update
                cv_factory.tell_pop(fitness)

        # ---------------------------------------------------------
        # Best module and final extraction
        # ---------------------------------------------------------
        module = cv_factory.best_module()

        feats_np = [module.get_output(img, return_map=return_maps) for img in images]
        feats_np = np.stack(feats_np, axis=0)  # [N, C, H, W] or [N, H, W]

        # ---------------------------------------------------------
        # Feature pruning and normalization
        # ---------------------------------------------------------
        if prune_features is not None:
            if feats_np.ndim == 4:  # [N, C, H, W]
                feats_np = feats_np[:, prune_features, :, :]
            else:
                feats_np = feats_np[prune_features, :, :]

        print_debugging(
            f"Convolution module features shape: {feats_np.shape}, "
            f"value range: {feats_np.min()} to {feats_np.max()}, var: {feats_np.var()}"
        )
        feats_np = (feats_np - feats_np.mean()) / (feats_np.std() + 1e-8)
        return feats_np
    
    def get_cv_module(
        self, images, cv_factory, return_maps=True, n_jobs=None, verbose=0, prune_features=None
    ):
        """
        Torchrun-friendly version with multiprocessing image parallelism.
        - Each rank processes a disjoint shard of images.
        - Each shard is processed in parallel using multiprocessing (per-image chunks).
        - Results are gathered with torch.distributed all_gather on tensors.
        - Output shape: [N_images, C, H, W] or [N_images, H, W] depending on your modules.
        """
        import multiprocessing as mp
        from joblib import Parallel, delayed
        import torch.distributed as dist
        import numpy as np
        import os

        # ---------------------------------------------------------
        # Normalize input to numpy
        # ---------------------------------------------------------
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = np.expand_dims(images, 0)  # [1, H, W]
        total_images = len(images)
        

        # ---------------------------------------------------------
        # Optimize module via variance (if no pruning provided)
        # ---------------------------------------------------------
        if prune_features is None:
            # ---------------------------------------------------------
            # CMA-ES optimization loop
            # ---------------------------------------------------------
            # Multiprocessing parallelization over MODULES
            modules = cv_factory.ask_pop()
            n_procs = min(n_jobs, len(modules))
            print_info(f"⚙️ Using {n_procs} processes for convolution module evaluation...")
            for epoch in trange(self.cv_epochs, desc="CV Module Optimization", disable=verbose == 0):
                print_debugging(f"Epoch {epoch}")
                modules = cv_factory.ask_pop()
                if not isinstance(modules, list):
                    modules = [modules]               

                fitness = self.eval_cv_pop(modules, images)

                print_debugging(f"Module fitness values: {fitness}")
                # CMA-ES update
                cv_factory.tell_pop(fitness)

        # ---------------------------------------------------------
        # Best module and final extraction
        # ---------------------------------------------------------
        module = cv_factory.best_module()

        return module



    def get_feature_tensor(self, images, prune_features=None):
        if self.feature_extractor == "handcrafted":
            return self.get_handcrafted_features(images, n_jobs=self.n_cpus, prune_features_idx=prune_features)
        elif self.feature_extractor == "cv_modules":
            if self.cv_factory is None:
                self.cv_factory = CVModuleFactory(
                    Optimizer={"class_name": "CMAES", "kwargs": {"n_params": 128, "lambda_": self.n_cv_modules, "seed": self.seed}},
                    CVModule={"class_name": "ConvolutionModule", "kwargs": {"n_filters": self.n_filters, "filter_sizes": 3, "device": self.device}}
                )
            return self.convolution_modules_tensor(images, self.cv_factory, return_maps=True, n_jobs=self.n_cpus, prune_features=prune_features)

        else:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
    
    def init_cv_module(self, images):
        self.cv_factory = CVModuleFactory(
                    Optimizer={"class_name": "CMAES", "kwargs": {"n_params": 128, "lambda_": self.n_cv_modules, "seed": self.seed}},
                    CVModule={"class_name": "ConvolutionModule", "kwargs": {"n_filters": self.n_filters, "filter_sizes": 3, "device": self.device}}
                )
        return self.get_cv_module(images, self.cv_factory, return_maps=True, n_jobs=self.n_cpus)
        
    def set_criterion(self, target):
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        pos_weight = self.compute_pos_weight(target)
        self.criterion = DiceCELoss(pos_weight=pos_weight.float().to(self.device))
    # --------------------------------------------------------
    # Training & Evaluation
    # --------------------------------------------------------
    def train(self, model, epochs=10, lr=1e-3, n_cpus=None):
        """
        Train PCNet model with unified threading, validation, and logging.
        Keeps print_info() calls for continuity.
        """
        # --------------------------------------------------------
        # Environment setup
        # --------------------------------------------------------
        self._setup_cpu_env()
        device = getattr(model, "device", "cpu")
        print_info(f"🧮 Using device: {device}")

        # --------------------------------------------------------
        # Count learnable parameters
        # --------------------------------------------------------

        # Collect parameters from all nodes
        
        print_info(f"🔢 Model has {len(model.params)} learnable parameter tensors.")

        optimizer = torch.optim.Adam(model.params, lr=lr, weight_decay=1e-4)

        # --------------------------------------------------------
        # Prepare data
        # --------------------------------------------------------
        feature_tensor = torch.from_numpy(self.feature_tensor).float()
        target = torch.from_numpy(self.train_masks).long()
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        train_loader = self.get_dataloaders(
            TensorDataset(feature_tensor, target),
            shuffle=True,
            batch_size=self.batch_size
        )

        val_loader = None
        if getattr(self, "val_feature_tensor", None) is not None and getattr(self, "val_masks", None) is not None:
            val_feature_tensor = torch.from_numpy(self.val_feature_tensor).float()
            val_target = torch.from_numpy(self.val_masks).long()
            if val_target.dim() == 4 and val_target.size(1) == 1:
                val_target = val_target.squeeze(1)
            val_loader = self.get_dataloaders(
                TensorDataset(val_feature_tensor, val_target),
                shuffle=False,
                batch_size=self.batch_size
            )

        # --------------------------------------------------------
        # Logs
        # --------------------------------------------------------
        train_losses, val_losses = [], []
        train_dices, val_dices = [], []
        train_ious, val_ious = [], []
        train_clious, val_clious = [], []
        gates_over_epochs = []

        # --------------------------------------------------------
        # Epoch loop
        # --------------------------------------------------------
        pbar = tqdm.tqdm(range(epochs), desc="Training PCNet")

        for epoch in pbar:
            batch_losses, batch_dices, batch_ious, batch_clious = [], [], [], []

            for ft_batch, t_batch in train_loader:
                ft_batch, t_batch = ft_batch.to(device), t_batch.to(device)

                optimizer.zero_grad()
                logits = model.evaluate(ft_batch)  # (B, C, H, W)
                loss = self.criterion(logits, t_batch)
                loss.backward()
                optimizer.step()

                # Metrics
                with torch.no_grad():
                    pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
                    true_mask = t_batch.unsqueeze(1).float()

                    dice_score = self.dice_coef(logits, t_batch)
                    iou_score = self.iou_score(logits, t_batch)
                    cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

                batch_losses.append(loss.item())
                batch_dices.append(float(torch.as_tensor(dice_score).mean()))
                batch_ious.append(float(torch.as_tensor(iou_score).mean()))
                batch_clious.append(float(torch.as_tensor(cl_iou_score).mean()))

            # Collect gate values (if present)
            gates_values = []
            for n in model.get_nodes():
                if hasattr(n, "gate"):
                    with torch.no_grad():
                        gates_values.append(float(torch.sigmoid(n.gate).item()))
            if gates_values:
                gates_over_epochs.append(gates_values)

            # Epoch metrics
            epoch_loss = np.mean(batch_losses) if batch_losses else 0.0
            epoch_dice = np.mean(batch_dices) if batch_dices else 0.0
            epoch_iou = np.mean(batch_ious) if batch_ious else 0.0
            epoch_cl_iou = np.mean(batch_clious) if batch_clious else 0.0

            train_losses.append(epoch_loss)
            train_dices.append(epoch_dice)
            train_ious.append(epoch_iou)
            train_clious.append(epoch_cl_iou)

            print_info(f"Epoch [{epoch+1}/{epochs}] — Train loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, CL-IoU: {epoch_cl_iou:.4f}")

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------
            if val_loader is not None:
                val_loss, val_dice, val_iou, val_cl_iou = self.evaluate(model, val_loader)
                val_losses.append(val_loss)
                val_dices.append(val_dice)
                val_ious.append(val_iou)
                val_clious.append(val_cl_iou)

                print_info(f"Validation — Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, CL-IoU: {val_cl_iou:.4f}")

            # Progress bar
            pbar.set_postfix({
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{val_losses[-1]:.4f}" if val_loader and val_losses else None
            })

            # Report to file every 5 epochs
            if (epoch + 1) % 5 == 0:
                with open(self.experiment_report, "a") as f:
                    f.write(f"{epoch+1}\ttrain\t{epoch_loss:.4f}\t{epoch_cl_iou:.4f}\n")
                    if val_loader:
                        f.write(f"\tval\t{val_losses[-1]:.4f}\t{val_clious[-1]:.4f}\n")

        # --------------------------------------------------------
        # Save logs to CSV
        # --------------------------------------------------------
        df_train = pd.DataFrame({
            "train_loss": train_losses,
            "train_dice": train_dices,
            "train_iou": train_ious,
            "train_cl_iou": train_clious
        })
        df_train.to_csv(os.path.join(self.log_path, "training.csv"), index=False)

        if val_loader:
            df_val = pd.DataFrame({
                "val_loss": val_losses,
                "val_dice": val_dices,
                "val_iou": val_ious,
                "val_cl_iou": val_clious
            })
            df_val.to_csv(os.path.join(self.log_path, "validation.csv"), index=False)

        # --------------------------------------------------------
        # Save gates (if any)
        # --------------------------------------------------------
        if gates_over_epochs:
            max_len = max(len(g) for g in gates_over_epochs)
            gates_array = np.array([
                np.pad(g, (0, max_len - len(g)), constant_values=np.nan)
                for g in gates_over_epochs
            ])
            df_gates = pd.DataFrame(gates_array, columns=[f"gate_{i}" for i in range(gates_array.shape[1])])
            df_gates.to_csv(os.path.join(self.log_path, "gates.csv"), index=False)

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------
        import matplotlib.pyplot as plt

        def _plot_metric(train, val, title, fname):
            plt.figure(figsize=(8, 5))
            plt.plot(train, label="Train")
            if val_loader:
                plt.plot(val, label="Val")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_path, fname))
            plt.close()

        _plot_metric(train_losses, val_losses, "Training Loss over Epochs", "training_plot.png")
        _plot_metric(train_dices, val_dices, "Dice over Epochs", "dice_plot.png")
        _plot_metric(train_ious, val_ious, "IoU over Epochs", "iou_plot.png")
        _plot_metric(train_clious, val_clious, "CL-IoU over Epochs", "cliou_plot.png")

        # --------------------------------------------------------
        # Final summary
        # --------------------------------------------------------
        print_info("✅ Training complete.")
        self.experiment_dict["training"] = {
            "final_loss": train_losses[-1],
            "final_dice": train_dices[-1],
            "final_iou": train_ious[-1],
            "final_cl_iou": train_clious[-1]
        }
        with open(self.experiment_report, "a") as f:
            f.write(f"Final Training — Loss: {train_losses[-1]:.4f}, Dice: {train_dices[-1]:.4f}, IoU: {train_ious[-1]:.4f}, CL-IoU: {train_clious[-1]:.4f}\n")
            if val_loader:
                f.write(f"Final Validation — Loss: {val_losses[-1]:.4f}, Dice: {val_dices[-1]:.4f}, IoU: {val_ious[-1]:.4f}, CL-IoU: {val_clious[-1]:.4f}\n")

        return model


    @torch.no_grad()
    def evaluate(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []
        for ft_batch, t_batch in dataloader:
            logits = model.evaluate(ft_batch)
            loss = self.criterion(logits, t_batch)
            dice_score = self.dice_coef(logits, t_batch)
            iou_score = self.iou_score(logits, t_batch)
            pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            true_mask = t_batch.unsqueeze(1).float()
            cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

            losses.append(loss.item())
            dices.append(float(torch.as_tensor(dice_score).mean()))
            ious.append(float(torch.as_tensor(iou_score).mean()))
            clious.append(float(torch.as_tensor(cl_iou_score)))

        return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious)), float(np.mean(clious))
    
    @torch.no_grad()
    def test(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        for ft_batch, t_batch in dataloader:
            logits = model.evaluate(ft_batch)
            loss = self.criterion(logits, t_batch)
            dice_score = self.dice_coef(logits, t_batch)
            iou_score = self.iou_score(logits, t_batch)
            pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            true_mask = t_batch.unsqueeze(1).float()
            cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

            # Flatten predictions and targets
            pm = pred_mask.view(-1).long()
            tm = true_mask.view(-1).long()

            tp = ((pm == 1) & (tm == 1)).sum()
            fp = ((pm == 1) & (tm == 0)).sum()
            fn = ((pm == 0) & (tm == 1)).sum()
            tn = ((pm == 0) & (tm == 0)).sum()

            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp + 1e-8)

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn + 1e-8)

            # F1 = harmonic mean
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)

            losses.append(loss.item())
            dices.append(float(torch.as_tensor(dice_score).mean()))
            ious.append(float(torch.as_tensor(iou_score).mean()))
            clious.append(float(torch.as_tensor(cl_iou_score)))

            accuracies.append(accuracy.item())
            precisions.append(precision.item())
            recalls.append(recall.item())
            f1_scores.append(f1_score.item())

        print_info("✅ Testing complete.")
        print_info(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}")
        print_info(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}")
        self.experiment_dict["results"] = {
            "test_loss": np.mean(losses),
            "test_dice": np.mean(dices),
            "test_iou": np.mean(ious),
            "test_cl_iou": np.mean(clious),
            "test_accuracy": np.mean(accuracies),
            "test_precision": np.mean(precisions),
            "test_recall": np.mean(recalls),
            "test_f1_score": np.mean(f1_scores)
        }
        n_trainable_params = sum(p.numel() for p in model.params if p.requires_grad)
        self.experiment_dict["n_trainable_params"] = n_trainable_params
        with open(self.experiment_report, "a") as f:
            f.write(f"Test Loss: {np.mean(losses):.4f}, Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}, CL IoU: {np.mean(clious):.4f}\n")
            f.write(f"Test Accuracy: {np.mean(accuracies):.4f}, Precision: {np.mean(precisions):.4f}, Recall: {np.mean(recalls):.4f}, F1-Score: {np.mean(f1_scores):.4f}\n")
        return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious)), float(np.mean(clious))

    # --------------------------------------------------------
    # I/O
    # --------------------------------------------------------
    def show_prediction(self, model):
        with open(self.experiment_report, "a") as f:
            f.write(str(model) + "\n")

        predictions_path = os.path.join(self.log_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        imgs = [self.test_data[i][0] for i in range(len(self.test_data))]

        # Process images in groups of 3
        max_pred = 15
        group_size = 3
        for j in range(0, max_pred):
            start_idx = random.randint(0, len(self.test_inputs) - group_size)
            end_idx = min(start_idx + group_size, len(self.test_inputs))
            group = list(zip(
                imgs[start_idx:end_idx],
                self.test_inputs[start_idx:end_idx],
                self.test_ground_truth[start_idx:end_idx]
            ))
            fig, axes = plt.subplots(3, group_size, figsize=(4 * group_size, 9))

            for col, (img, inputs, gt) in enumerate(group):
                mask = model.predict(inputs)
                img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
                gt_np = gt.squeeze().cpu().numpy()
                mask_np = mask[0][0] if isinstance(mask, (list, tuple)) else mask.squeeze()

                # Row 1 → Input
                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f"Input {start_idx + col}")
                axes[0, col].axis("off")

                # Row 2 → Ground Truth
                axes[1, col].imshow(gt_np, cmap="gray")
                axes[1, col].set_title(f"Ground Truth {start_idx + col}")
                axes[1, col].axis("off")

                # Row 3 → Prediction
                axes[2, col].imshow(mask_np, cmap="cool")
                axes[2, col].set_title(f"Prediction {start_idx + col}")
                axes[2, col].axis("off")

            # Hide any unused columns (in case last group < 3)
            for col in range(len(group), group_size):
                axes[0, col].axis("off")
                axes[1, col].axis("off")
                axes[2, col].axis("off")

            plt.savefig(os.path.join(predictions_path, f"predictions_{start_idx//group_size}.png"), bbox_inches="tight")
            plt.close(fig)

        print_debugging(self.log_path)
        # model.visualize(save_path=self.log_path)

    def finetune(self):
        pass

    def save_model(self, model_name="pcnet"):
        with open(os.path.join(self.log_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.save_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, model_name="pcnet"):
        with open(model_name, "rb") as f:
            self.model = pickle.load(f)

    # --------------------------------------------------------
    # Orchestration
    # --------------------------------------------------------
    def run(self):
        self._setup_cpu_env()
        self.set_dataset(self.data_path, self.test_path)
        print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            self.feature_tensor = self.get_feature_tensor(np.array([self.train_data[i][0] for i in range(len(self.train_data))]))
            self.train_masks = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
            self.val_feature_tensor = self.get_feature_tensor(np.array([self.val_data[i][0] for i in range(len(self.val_data))]))
            self.val_masks = np.array([self.val_data[i][1] for i in range(len(self.val_data))]) if self.validation_split > 0 else None
            self.set_criterion(torch.from_numpy(self.train_masks).float().to(self.device))

            self.model.init_network(self.feature_tensor, self.train_masks)
            with open(self.experiment_report, "a") as f:
                f.write(str(self.model) + "\n")
            if self.device == "cuda":
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            else:
                self.model = self.train_parallel(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            self.save_model()

            if self.test_path is not None:
                self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)

                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth), shuffle=False, batch_size=self.batch_size)
                loss, dice, iou, cl_iou = self.evaluate(self.model, test_loader)
                print_info("Test Loss:", loss, "Dice:", dice, "IoU:", iou, "CL IoU:", cl_iou)
                with open(self.experiment_report, "a") as f:
                    f.write(f"Test Loss: {loss}, Dice: {dice}, IoU: {iou}, CL IoU: {cl_iou}\n")
                self.show_prediction(self.model)
        elif self.execution_mode == "test":
            self.load_model(self.model_name)
            self.test_inputs = torch.from_numpy(self.get_feature_tensor(np.array([self.test_data[i][0] for i in range(len(self.test_data))]))).float().to(self.device)
            self.test_ground_truth = torch.from_numpy(np.array([self.test_data[i][1] for i in range(len(self.test_data))])).long().to(self.device)
            if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                self.test_ground_truth = self.test_ground_truth.squeeze(1)
            self.show_prediction(self.model)


class ExperimentGEPC(ExperimentPC):
    def __init__(self, config):
        super().__init__(config)
        self.ge_config = self.config.get("ge", {"generations": 10, "pop_size": 8, "genome_len": 96, "crossover_rate": 0.9, "mutation_rate": 0.05, "elite_k": 2, "train_split": 0.05})
        self.ge = GEOptimizer(
                    pop_size=self.ge_config["pop_size"],
                    genome_len=self.ge_config["genome_len"],
                    max_depth=self.max_depth,          # search budget
                    max_branching=self.max_branching,      # upper bound for sum fan-in
                    crossover_rate=self.ge_config["crossover_rate"],
                    mutation_rate=self.ge_config["mutation_rate"],
                    elite_k=self.ge_config["elite_k"],
                    n_classes=self.n_classes,
                    device=self.device,
                    n_jobs=self.n_cpus,
                    report=self.experiment_report,
                    log_path = self.log_path
                )
        
    def get_prune_features_idx(self, model):
        feats = []
        new_name = 0
        for n in model.get_nodes():
            if isinstance(n, InputNode):
                feats.append(n.feature_idx)
                n.feature_name = f"feat_{new_name}"
                n.feature_idx = new_name
                new_name += 1

        return feats
    
    def worker_cv_module_process(self, model, module, X_np, Y_t, device, prune_idx, queue):
        """
        Worker that evaluates one or more modules on all training images.
        Returns list of fitness values (one per module).
        """
            
        # Extract features for this module
        feats = []
        for img in X_np:
            out = module.get_output(img, return_map=True)
            feats.append(torch.from_numpy(out))
        X_tensor = torch.stack(feats).float().to(device)

        if prune_idx is not None:
            X_tensor = X_tensor[:, prune_idx, :, :]
    
        dataset = TensorDataset(X_tensor, Y_t)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        loss_total, cliou_total = 0.0, 0.0
        with torch.no_grad():
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                logits = model.evaluate(Xb)
                loss_total += float(self.criterion(logits, Yb.squeeze(1)).item())
                cliou_total += float(
                    self.cl_iou_score(
                        torch.argmax(logits, dim=1, keepdim=True).float(), Yb
                    )
                )
        avg_loss = loss_total / max(1, len(loader))
        return avg_loss
    
    def train_cv_modules(self, model, X_train, Y_train, prune_features=None, epochs=10, lr=1e-2, n_cpus=8):
        self.cv_factory._optimizer.best = None  # reset best found solution
        self.cv_factory._optimizer.best_fitness = -np.inf  # reset best found solution

        for epoch in trange(epochs, desc="Training CV Modules"):
            print_info(f"CV Modules Training Epoch {epoch + 1}/{epochs}")
            modules = self.cv_factory.ask_pop()
            if not isinstance(modules, list):
                modules = [modules]

            # Split modules into chunks for each worker
            n_procs = min(n_cpus, len(modules))
            with multiprocessing.Pool(n_procs) as pool:
                fitnesses = pool.starmap(
                    self.worker_cv_module_process,
                    [(model, module, X_train, Y_train, self.device, prune_features, None) for module in modules]
                )
            print_info(f"   CV Module Fitnesses: {fitnesses}")
            self.cv_factory.tell_pop(fitnesses)

        return self.cv_factory.best_module()
        
    def run(self):
        
        self.set_dataset(self.data_path, self.test_path)
        # print_info("Dataset size:", len(self.data))
        if self.execution_mode == "train":
            self.set_algorithm()
            

            ge_train_size = int(self.ge_config["train_split"] * len(self.train_data))
            print_info("GE Training size:", ge_train_size)
            print_info(len(self.train_data))
            ge_data = []
            ge_masks = []
            for i in range(ge_train_size):
                try:
                    x = self.train_data[i]
                    ge_data.append(x[0])
                    ge_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing training sample {i}: {e}")
            ge_X_train = np.array(ge_data)
            ge_masks = np.array(ge_masks)
            ge_X_train = torch.from_numpy(ge_X_train).float().to(self.device)
            ge_Y_train = torch.from_numpy(ge_masks).long().to(self.device)
            self.set_criterion(ge_Y_train)
            best_pc, best_genome, history = self.ge.evolve(ge_X_train, ge_Y_train, criterion=self.criterion, generations = self.ge_config["generations"], n_classes=self.n_classes, train_steps=50, lr=self.learning_rate)
            print_info("Best genome:", best_genome)
            print_info("Best PC:", str(best_pc))
            with open(self.experiment_report, "a") as f:
                f.write(str(best_pc) + "\n")
                f.write(f"History: {history}\n")
                f.write(f"Best genome: {best_genome}\n")

            self.experiment_dict["ge"] = {}
            self.experiment_dict["ge"]["best_genome"] = best_genome
            self.experiment_dict["ge"]["history"] = history

            self.model = best_pc
            f.close()
            self.model.cv_module = None
            train_data = []
            train_masks = []
            for i in range(len(self.train_data)):
                try:
                    x = self.train_data[i]
                    train_data.append(x[0])
                    train_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing training sample {i}: {e}")

            self.feature_tensor = np.array(train_data)
            self.train_masks = np.array(train_masks)

            val_data = []
            val_masks = []
            for i in range(len(self.val_data)):
                try:
                    x = self.val_data[i]
                    val_data.append(x[0])
                    val_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing validation sample {i}: {e}")

            self.val_feature_tensor = np.array(val_data)
            self.val_masks = np.array(val_masks)
            self.set_criterion(torch.from_numpy(self.train_masks).long().to(self.device))
            if self.device == "cuda":
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            else:
                self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate, n_cpus=self.n_cpus)
            self.save_model("pcnet-ge")
            if self.test_path is not None:
                test_data = []
                test_masks = []
                for i in range(len(self.test_data)):
                    try:
                        x = self.test_data[i]
                        test_data.append(x[0])
                        test_masks.append(x[1])
                    except Exception as e:
                        print_error(f"Error processing validation sample {i}: {e}")
                self.test_inputs = torch.from_numpy(np.array(test_data)).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array(test_masks)).long().to(self.device)
                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)
                
                test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth), shuffle=False, batch_size=self.batch_size)
                loss, dice, iou, cl_iou = self.test(self.model, test_loader)
                json.dump(self.experiment_dict, open(os.path.join(self.config["log_dir"], "experiment.json"), "w"), indent=4)
                self.show_prediction(self.model)
        elif self.execution_mode == "test":
            self.model_path = self.config["model_path"]
            self.load_model(self.model_path)
            test_data = []
            test_masks = []
            for i in range(len(self.test_data)):
                try:
                    x = self.test_data[i]
                    test_data.append(x[0])
                    test_masks.append(x[1])
                except Exception as e:
                    print_error(f"Error processing validation sample {i}: {e}")
            self.test_inputs = torch.from_numpy(np.array(test_data)).float().to(self.device)
            self.test_ground_truth = torch.from_numpy(np.array(test_masks)).long().to(self.device)
            self.set_criterion(self.test_ground_truth)
            if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                self.test_ground_truth = self.test_ground_truth.squeeze(1)
            test_loader = self.get_dataloaders(TensorDataset(self.test_inputs, self.test_ground_truth), shuffle=False, batch_size=self.batch_size)
            loss, dice, iou, cl_iou = self.test(self.model, test_loader)
            json.dump(self.experiment_dict, open(os.path.join(self.config["log_dir"], "experiment.json"), "w"), indent=4)


class ExperimentUnsupervisedGEPC(ExperimentPC):
    def __init__(self, config):
        super().__init__(config)

        self.ge_config = self.config.get(
            "ge",
            {
                "generations": 10,
                "pop_size": 8,
                "genome_len": 96,
                "crossover_rate": 0.9,
                "mutation_rate": 0.05,
                "elite_k": 2,
                "train_split": 0.05,
            },
        )

        self.ge = UnsupervisedGEOptimizer(
            pop_size=self.ge_config["pop_size"],
            genome_len=self.ge_config["genome_len"],
            max_depth=self.max_depth,
            max_branching=self.max_branching,
            crossover_rate=self.ge_config["crossover_rate"],
            mutation_rate=self.ge_config["mutation_rate"],
            elite_k=self.ge_config["elite_k"],
            n_classes=self.n_classes,
            device=self.device,
            n_jobs=self.n_cpus,
            report=self.experiment_report,
            log_path=self.log_path,
        )

    # --------------------------------------------------------
    # SET UNSUPERVISED CRITERION
    # --------------------------------------------------------
    def set_criterion(self):
        self.criterion = UnsupervisedPCNetLoss()

    # --------------------------------------------------------
    # TRAIN (uses only X for loss, Y for metrics)
    # --------------------------------------------------------
    def train(self, model, epochs=10, lr=1e-3, n_cpus=None):
        """
        Train PCNet model (UNSUPERVISED loss, supervised metrics).
        Loss = criterion(logits, ft_batch)
        Metrics use t_batch only for evaluation.
        """
        self._setup_cpu_env()
        device = model.device

        print_info(f"🧮 Using device: {device}")
        print_info(f"🔢 Model has {len(model.params)} learnable parameter tensors.")

        optimizer = torch.optim.Adam(model.params, lr=lr, weight_decay=1e-4)

        # ------------------------------
        # Data preparation
        # ------------------------------
        X = torch.from_numpy(self.feature_tensor).float()
        Y = torch.from_numpy(self.train_masks).long()

        if Y.dim() == 4 and Y.size(1) == 1:
            Y = Y.squeeze(1)

        train_loader = self.get_dataloaders(
            TensorDataset(X, Y),
            shuffle=True,
            batch_size=self.batch_size,
        )

        val_loader = None
        if hasattr(self, "val_feature_tensor") and self.val_feature_tensor is not None:
            Xv = torch.from_numpy(self.val_feature_tensor).float()
            Yv = torch.from_numpy(self.val_masks).long()
            if Yv.dim() == 4 and Yv.size(1) == 1:
                Yv = Yv.squeeze(1)

            val_loader = self.get_dataloaders(
                TensorDataset(Xv, Yv),
                shuffle=False,
                batch_size=self.batch_size,
            )

        # ------------------------------
        # Training loop
        # ------------------------------
        train_losses, val_losses = [], []
        train_dices, val_dices = [], []
        train_ious, val_ious = [], []
        train_clious, val_clious = [], []
        gates_over_epochs = []

        pbar = tqdm.tqdm(range(epochs), desc="Training PCNet (unsupervised loss)")

        for epoch in pbar:
            batch_losses, batch_dices, batch_ious, batch_clious = [], [], [], []

            for ft_batch, t_batch in train_loader:
                ft_batch = ft_batch.to(device)
                t_batch = t_batch.to(device)

                optimizer.zero_grad()

                logits = model.evaluate(ft_batch)
                # UNSUPERVISED loss
                loss = self.criterion(logits, ft_batch)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred_mask = torch.argmax(logits, 1, keepdim=True).float()
                    print_debugging(pred_mask.unique(return_counts=True))
                    true_mask = t_batch.unsqueeze(1).float()

                    dice_score = self.dice_coef(logits, t_batch)
                    iou_score = self.iou_score(logits, t_batch)
                    cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

                batch_losses.append(loss.item())
                batch_dices.append(float(dice_score))
                batch_ious.append(float(iou_score))
                batch_clious.append(float(cl_iou_score))

            epoch_loss = np.mean(batch_losses)
            epoch_dice = np.mean(batch_dices)
            epoch_iou = np.mean(batch_ious)
            epoch_cl_iou = np.mean(batch_clious)

            train_losses.append(epoch_loss)
            train_dices.append(epoch_dice)
            train_ious.append(epoch_iou)
            train_clious.append(epoch_cl_iou)

            print_info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Dice={epoch_dice:.4f}")

            if val_loader is not None:
                val_loss, val_dice, val_iou, val_cl_iou = self.evaluate(model, val_loader)
                val_losses.append(val_loss)
                val_dices.append(val_dice)
                val_ious.append(val_iou)
                val_clious.append(val_cl_iou)

                print_info(f"Validation Loss={val_loss:.4f}, Dice={val_dice:.4f}")

            pbar.set_postfix({"loss": epoch_loss})

        # Save CSV logs, plots, etc. (unchanged)
        # super()._finalize_training_logs(
        #     train_losses, train_dices, train_ious, train_clious,
        #     val_losses, val_dices, val_ious, val_clious, gates_over_epochs
        # )

        print_info("✅ Training complete.")
        return model
    
    @torch.no_grad()
    def evaluate(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []

        for ft_batch, t_batch in dataloader:
            ft_batch = ft_batch.to(self.device).float()    # PCNet expects float features
            t_batch = t_batch.to(self.device).long()       # masks for metrics only

            # -------------------------------------------------------
            # Forward pass (PCNet requires evaluate(), not model(x))
            # -------------------------------------------------------
            logits = model.evaluate(ft_batch)   # (B, C, H, W) in log-space

            # -------------------------------------------------------
            # Unsupervised loss uses **features**, NOT labels
            # -------------------------------------------------------
            loss = self.criterion(logits, ft_batch)
            losses.append(loss.item())

            # -------------------------------------------------------
            # Supervised metrics for monitoring only
            # -------------------------------------------------------
            dice_score = self.dice_coef(logits, t_batch)
            iou_score  = self.iou_score(logits, t_batch)

            pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            true_mask = t_batch.unsqueeze(1).float()

            cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

            dices.append(float(dice_score))
            ious.append(float(iou_score))
            clious.append(float(cl_iou_score))

        return (
            float(np.mean(losses)),
            float(np.mean(dices)),
            float(np.mean(ious)),
            float(np.mean(clious)),
        )

    @torch.no_grad()
    def test(self, model, dataloader):
        losses, dices, ious, clious = [], [], [], []
        accuracies, precisions, recalls, f1_scores = [], [], [], []

        for ft_batch, t_batch in dataloader:
            ft_batch = ft_batch.to(self.device).float()   # features for loss
            t_batch = t_batch.to(self.device).long()      # masks for metrics

            # ---------------------------------------------------------
            # Forward pass
            # ---------------------------------------------------------
            logits = model.evaluate(ft_batch)

            # ---------------------------------------------------------
            # UNSUPERVISED LOSS (depends ONLY on images)
            # ---------------------------------------------------------
            loss = self.criterion(logits, ft_batch)
            losses.append(loss.item())

            # ---------------------------------------------------------
            # SUPERVISED METRICS (compare vs. masks)
            # ---------------------------------------------------------
            dice_score = self.dice_coef(logits, t_batch)
            iou_score = self.iou_score(logits, t_batch)

            pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            true_mask = t_batch.unsqueeze(1).float()

            cl_iou_score = self.cl_iou_score(pred_mask, true_mask)

            dices.append(float(dice_score))
            ious.append(float(iou_score))
            clious.append(float(cl_iou_score))

            # ---------------------------------------------------------
            # Classification metrics per pixel
            # ---------------------------------------------------------
            pm = pred_mask.view(-1).long()
            tm = true_mask.view(-1).long()

            tp = ((pm == 1) & (tm == 1)).sum()
            fp = ((pm == 1) & (tm == 0)).sum()
            fn = ((pm == 0) & (tm == 1)).sum()
            tn = ((pm == 0) & (tm == 0)).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

            accuracies.append(float(accuracy))
            precisions.append(float(precision))
            recalls.append(float(recall))
            f1_scores.append(float(f1_score))

        # ---------------------------------------------------------
        # Aggregate results
        # ---------------------------------------------------------
        mean_loss = float(np.mean(losses))
        mean_dice = float(np.mean(dices))
        mean_iou = float(np.mean(ious))
        mean_cl_iou = float(np.mean(clious))
        mean_acc = float(np.mean(accuracies))
        mean_prec = float(np.mean(precisions))
        mean_rec = float(np.mean(recalls))
        mean_f1 = float(np.mean(f1_scores))

        # ---------------------------------------------------------
        # Print nicely
        # ---------------------------------------------------------
        print_info("✅ Testing complete.")
        print_info(
            f"Test Loss: {mean_loss:.4f}, Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, "
            f"CL-IoU: {mean_cl_iou:.4f}"
        )
        print_info(
            f"Test Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, "
            f"Recall: {mean_rec:.4f}, F1-Score: {mean_f1:.4f}"
        )

        # ---------------------------------------------------------
        # Save to experiment_dict
        # ---------------------------------------------------------
        self.experiment_dict["results"] = {
            "test_loss": mean_loss,
            "test_dice": mean_dice,
            "test_iou": mean_iou,
            "test_cl_iou": mean_cl_iou,
            "test_accuracy": mean_acc,
            "test_precision": mean_prec,
            "test_recall": mean_rec,
            "test_f1_score": mean_f1,
        }

        # ---------------------------------------------------------
        # Also save #params info
        # ---------------------------------------------------------
        n_trainable_params = sum(
            p.numel() for p in model.params if p.requires_grad
        )
        self.experiment_dict["n_trainable_params"] = n_trainable_params

        # ---------------------------------------------------------
        # Log to file
        # ---------------------------------------------------------
        with open(self.experiment_report, "a") as f:
            f.write(
                f"Test Loss: {mean_loss:.4f}, Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, "
                f"CL-IoU: {mean_cl_iou:.4f}\n"
            )
            f.write(
                f"Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, "
                f"Recall: {mean_rec:.4f}, F1: {mean_f1:.4f}\n"
            )

        return mean_loss, mean_dice, mean_iou, mean_cl_iou



    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------
    def run(self):

        # Load training/validation/test sets
        self.set_dataset(self.data_path, self.test_path)

        if self.execution_mode == "train":
            self.set_algorithm()
            self.set_criterion()

            # ------------------------------
            # GE SEARCH DATASET
            # ------------------------------
            ge_train_size = int(self.ge_config["train_split"] * len(self.train_data))
            print_info(f"GE Training size: {ge_train_size}")

            ge_data = []
            for i in range(ge_train_size):
                x, _ = self.train_data[i]       # ignore masks
                ge_data.append(x)

            ge_X_train = torch.from_numpy(np.array(ge_data)).float().to(self.device)

            # ------------------------------
            # RUN GE EVOLUTION (UNSUPERVISED)
            # ------------------------------
            best_pc, best_genome, history = self.ge.evolve(
                ge_X_train,
                criterion=self.criterion,
                generations=self.ge_config["generations"],
                n_classes=self.n_classes,
                train_steps=50,
                lr=self.learning_rate,
            )

            print_info("Best genome:", best_genome)
            print_info("Best PC:", str(best_pc))

            self.experiment_dict["ge"] = {
                "best_genome": best_genome,
                "history": history
            }

            self.model = best_pc

            # ------------------------------
            # FULL TRAIN DATASET FOR PCNet TRAINING
            # ------------------------------
            train_data = [x for (x, y) in self.train_data]
            train_masks = [y for (x, y) in self.train_data]

            self.feature_tensor = np.array(train_data)
            self.train_masks = np.array(train_masks)

            val_data = [x for (x, y) in self.val_data]
            val_masks = [y for (x, y) in self.val_data]

            self.val_feature_tensor = np.array(val_data)
            self.val_masks = np.array(val_masks)

            # DO NOT PASS LABELS TO CRITERION ANYMORE
            self.set_criterion()

            # ------------------------------
            # TRAIN FINAL PCNet
            # ------------------------------
            self.model = self.train(self.model, epochs=self.epochs, lr=self.learning_rate)

            # Save
            self.save_model("pcnet-ge")

            # ------------------------------
            # TEST
            # ------------------------------
            if self.test_path is not None:
                test_data = [x for (x, y) in self.test_data]
                test_masks = [y for (x, y) in self.test_data]

                self.test_inputs = torch.from_numpy(np.array(test_data)).float().to(self.device)
                self.test_ground_truth = torch.from_numpy(np.array(test_masks)).long().to(self.device)

                if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                    self.test_ground_truth = self.test_ground_truth.squeeze(1)

                test_loader = self.get_dataloaders(
                    TensorDataset(self.test_inputs, self.test_ground_truth),
                    shuffle=False, batch_size=self.batch_size
                )

                loss, dice, iou, cl_iou = self.test(self.model, test_loader)

                json.dump(
                    self.experiment_dict,
                    open(os.path.join(self.config["log_dir"], "experiment.json"), "w"),
                    indent=4
                )

                self.show_prediction(self.model)

        # --------------------------------------------------------
        # TEST MODE
        # --------------------------------------------------------
        elif self.execution_mode == "test":
            self.load_model(self.config["model_path"])

            test_data = [x for (x, y) in self.test_data]
            test_masks = [y for (x, y) in self.test_data]

            self.test_inputs = torch.from_numpy(np.array(test_data)).float().to(self.device)
            self.test_ground_truth = torch.from_numpy(np.array(test_masks)).long().to(self.device)

            if self.test_ground_truth.dim() == 4 and self.test_ground_truth.size(1) == 1:
                self.test_ground_truth = self.test_ground_truth.squeeze(1)

            self.set_criterion()

            test_loader = self.get_dataloaders(
                TensorDataset(self.test_inputs, self.test_ground_truth),
                shuffle=False, batch_size=self.batch_size,
            )

            loss, dice, iou, cl_iou = self.test(self.model, test_loader)

            json.dump(
                self.experiment_dict,
                open(os.path.join(self.config["log_dir"], "experiment.json"), "w"),
                indent=4
            )
