import json
import cv2
import numpy as np

# For removing BG

from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import os
import pandas as pd
from torchvision import transforms

import os, json, cv2
import numpy as np
import pandas as pd

import os, json, cv2
import numpy as np
import pandas as pd

def from_csv_to_multiclass_masks(
    csv_path,
    image_dir,
    output_mask_dir,
    output_overlay_dir,
    label_map=None,
):
    """
    Convert polygon annotations from Label Studio-like CSV into multi-class masks.
    Works with .png, .jpg, .jpeg image files.
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_overlay_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    all_labels = set()

    # ---------- PASS 1: collect all label names ----------
    for _, row in df.iterrows():
        anns = row.get("annotations")
        if pd.isna(anns):
            continue
        try:
            ann_list = json.loads(anns)
        except Exception:
            try:
                ann_list = json.loads(anns.replace("'", "\""))
            except Exception:
                continue

        for ann in ann_list:
            val = ann.get("value") or ann.get("Value")
            if not isinstance(val, dict):
                try:
                    val = json.loads(val)
                except Exception:
                    continue
            if ann.get("type") == "polygonlabels" or ann.get("Type") == "polygonlabels":
                labels = val.get("polygonlabels", [])
                all_labels.update(labels)

    if label_map is None:
        label_map = {name: i + 1 for i, name in enumerate(sorted(all_labels))}
        print("Auto label map:", label_map)

    colors = {
        label: tuple(np.random.randint(0, 255, 3).tolist())
        for label in label_map.keys()
    }

    def find_image(base_name):
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            p = os.path.join(image_dir, base_name + ext)
            if os.path.exists(p):
                return p
        return None

    # ---------- PASS 2: create masks ----------
    for _, row in df.iterrows():
        base_name = os.path.splitext(str(row["id"]))[0]
        image_path = find_image(base_name)
        if not image_path:
            print(f"⚠️  Image not found for {base_name}")
            continue

        anns = row.get("annotations")
        if pd.isna(anns):
            continue

        try:
            ann_list = json.loads(anns)
        except Exception:
            try:
                ann_list = json.loads(anns.replace("'", "\""))
            except Exception:
                print(f"Invalid JSON for {base_name}")
                continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Cannot read image {image_path}")
            continue

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        overlay = image.copy()

        for ann in ann_list:
            val = ann.get("value") or ann.get("Value")
            if not isinstance(val, dict):
                try:
                    val = json.loads(val)
                except Exception:
                    continue

            if ann.get("type") != "polygonlabels" and ann.get("Type") != "polygonlabels":
                continue

            pts = val.get("points", [])
            labels = val.get("polygonlabels", [])
            if not pts or not labels:
                continue

            label = labels[0]
            
            if label not in label_map:
                continue

            class_id = label_map[label]

            # Handle both percent and absolute coordinate systems
            if max(p[0] for p in pts) <= 100 and max(p[1] for p in pts) <= 100:
                poly = np.array(
                    [[int(x / 100 * w), int(y / 100 * h)] for x, y in pts],
                    np.int32,
                )
            else:
                poly = np.array(pts, np.int32)
            cv2.fillPoly(mask, [poly], class_id)
            
            cv2.polylines(overlay, [poly], True, colors[label], 2)
            overlay_mask = mask == class_id
            overlay[overlay_mask] = (
                0.6 * overlay[overlay_mask] + 0.4 * np.array(colors[label])
            ).astype(np.uint8)

        mask_path = os.path.join(output_mask_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(output_overlay_dir, f"{base_name}_overlay.png")
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)

        print(f"✅ Saved mask & overlay for {base_name}")

    print("\n✅ Done! Masks and overlays saved to:")
    print(f" - Masks: {output_mask_dir}")
    print(f" - Overlays: {output_overlay_dir}")


def background_removal(image_dir, output_path):
    
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    os.makedirs(output_path, exist_ok=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cpu')
    birefnet.eval()
    birefnet.half()

    def extract_object(birefnet, imagepath):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(imagepath)
        input_images = transform_image(image).unsqueeze(0).to('cpu').half()

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image, mask

    images_files = os.listdir(image_dir)
    images_path = [os.path.join(image_dir, img) for img in images_files]

    for image_path in images_path:
        orig_im = io.imread(image_path)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        # process image
        pil_image = Image.open(image_path).convert("RGB")
        result_image = extract_object(birefnet, image_path)[0]

        # apply alpha mask
        pil_mask_im = Image.fromarray(result_image)
        orig_image = Image.open(image_path).convert("RGBA")
        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask_im)

        # build output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(output_path, f"{name}_no_bg.png")

        # save result
        no_bg_image.save(output_file)
        print(f"Saved: {output_file}")
        

if __name__ == "__main__":
    # Example usage:
    # from_json_to_mask("path/to/label_studio_export.json", "output/mask.png", "output/overlay.png", "path/to/original_image.png")
    output_dir = "data/A22/no_bg/images"
    # background_removal("data/A22/images", output_dir)
    print(f"Background removed images are saved in {output_dir}")
    from_csv_to_multiclass_masks("data/A22/stored_images.csv", "data/A22/images", "data/A22/masks", "data/A22/overlays")