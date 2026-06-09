import os
import json
import numpy as np
import nibabel as nib
import torch
import subprocess
import cv2

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_nifti_2d(img_hw: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    # (H,W) -> (H,W,1)
    vol = img_hw[..., None]
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float32)
    return nib.Nifti1Image(vol, affine)

import os

def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return True
    except Exception:
        return False



def resolve_nnunet_paths(config: dict, project_root: str = None) -> dict:
    """
    Ensures nnUNet_raw / nnUNet_preprocessed / nnUNet_results point to writable locations.
    If config paths are missing OR not writable, uses <project_root>/nnunet_data/*.
    """
    if project_root is None:
        # project_root = repo root assuming this file is in src/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    base = os.path.join(project_root, "nnunet_data")
    fallback = {
        "nnUNet_raw": os.path.join(base, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(base, "nnUNet_preprocessed"),
        "nnUNet_results": os.path.join(base, "nnUNet_results"),
    }

    for k, fb in fallback.items():
        p = config.get(k)
        if (p is None) or (not _is_writable_dir(p)):
            config[k] = fb
            os.makedirs(fb, exist_ok=True)

    return config


def export_crackdataset_subset_to_nnunet_raw(
    train_ds,                   # torch.utils.data.Dataset yielding (img, mask)
    test_ds,                    # Dataset yielding (img, mask) for evaluation; imagesTs needs only images
    nnunet_raw_dir: str,
    dataset_id: int,
    dataset_name: str,
    input_channels: int = 3,    # from your config
    spacing=(1.0, 1.0, 1.0),
):
    """
    Writes:
      nnUNet_raw/DatasetXXX_NAME/
        imagesTr/case_000000_0000.nii.gz ...
        labelsTr/case_000000.nii.gz ...
        imagesTs/case_100000_0000.nii.gz ...
        dataset.json
    """
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    base = os.path.join(nnunet_raw_dir, dataset_folder)
    imagesTr = os.path.join(base, "imagesTr")
    labelsTr = os.path.join(base, "labelsTr")
    imagesTs = os.path.join(base, "imagesTs")

    _ensure_dir(imagesTr)
    _ensure_dir(labelsTr)
    _ensure_dir(imagesTs)

    # ---- TRAIN ----
    for i in range(len(train_ds)):
        img, mask = train_ds[i]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # img: (C,H,W), mask: (1,H,W) or (H,W)
        if img.ndim != 3:
            raise ValueError(f"Expected img (C,H,W), got {img.shape}")
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected mask (H,W), got {mask.shape}")

        case_id = f"case_{i:06d}"

        # per-channel NIfTI
        C = img.shape[0]
        if C != input_channels:
            # not fatal, but usually indicates mismatch
            print(f"[WARN] input_channels={input_channels} but sample has C={C}")

        for c in range(C):
            nii = _to_nifti_2d(img[c].astype(np.float32), spacing=spacing)
            nib.save(nii, os.path.join(imagesTr, f"{case_id}_{c:04d}.nii.gz"))

        # binarize mask (your CrackDataset already outputs float 0/1)
        seg = (mask > 0.5).astype(np.uint8)
        seg_nii = _to_nifti_2d(seg, spacing=spacing)
        nib.save(seg_nii, os.path.join(labelsTr, f"{case_id}.nii.gz"))

    # ---- TEST imagesTs ----
    for j in range(len(test_ds)):
        img, _ = test_ds[j]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim != 3:
            raise ValueError(f"Expected img (C,H,W), got {img.shape}")

        case_id = f"case_{100000 + j:06d}"
        C = img.shape[0]
        for c in range(C):
            nii = _to_nifti_2d(img[c].astype(np.float32))
            nib.save(nii, os.path.join(imagesTs, f"{case_id}_{c:04d}.nii.gz"))

    # ---- dataset.json (nnUNet v2) ----
    channel_names = {str(i): f"channel_{i}" for i in range(input_channels)}
    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "crack": 1},
        "numTraining": int(len(train_ds)),
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    return base

def _run(cmd, env):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def nnunet_plan_preprocess(dataset_id: int, env):
    _run(["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"], env)

def nnunet_train(dataset_id: int, cfg: str, folds, trainer: str, plans: str,
                 num_gpus: int, env, device: str = None,
                 num_epochs: int = None, num_iterations_per_epoch: int = None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        cfg,
        str(folds),
        "-tr", trainer,
        "-p", plans,
        "-device", device,
    ]

    # pass trainer kwargs if provided
    trainer_kwargs = {}
    if num_epochs is not None:
        trainer_kwargs["num_epochs"] = int(num_epochs)
    if num_iterations_per_epoch is not None:
        trainer_kwargs["num_iterations_per_epoch"] = int(num_iterations_per_epoch)

    if trainer_kwargs:
        cmd += ["--trainer_kwargs", json.dumps(trainer_kwargs)]

    if device == "cuda":
        cmd += ["-num_gpus", str(num_gpus)]

    _run(cmd, env)

def nnunet_predict(dataset_id: int, in_dir: str, out_dir: str, cfg: str, folds, trainer: str, plans: str, env):
    _ensure_dir(out_dir)
    fold_args = []
    if isinstance(folds, (list, tuple)):
        fold_args = [str(f) for f in folds]
    else:
        fold_args = [str(folds)]
    _run([
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", in_dir,
        "-o", out_dir,
        "-c", cfg,
        "-tr", trainer,
        "-p", plans,
        "-f", *fold_args
    ], env)

import numpy as np
import nibabel as nib

def dice_iou_binary(pred01: np.ndarray, gt01: np.ndarray, eps=1e-8):
    pred01 = pred01.astype(np.uint8)
    gt01 = gt01.astype(np.uint8)
    inter = (pred01 & gt01).sum()
    dice = (2 * inter) / (pred01.sum() + gt01.sum() + eps)
    union = (pred01 | gt01).sum()
    iou = inter / (union + eps)
    return float(dice), float(iou)

