from datetime import datetime
import string
import numpy as np
from skimage import data, color
import cv2
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import os
import shutil

def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format dd-mm-yyyy_:mm:ss_<random string>
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[33m'
    RESULT =  '\033[94m'
    WARNING = '\033[93m'
    DEBUGGING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def now():
    return datetime.now().strftime("%H:%M:%S")

def print_error(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.FAIL+"ERROR"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_warning(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.ORANGE+"WARNING"+colors.ENDC+"] " +" ".join(map(str,args))+ "\n", **kwargs)

def print_debugging(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.DEBUGGING+"DEBUG"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_info(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.OKGREEN+"INFO"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)

def print_configs(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.OKBLUE+"CONFIGS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)
    
def print_results(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(now()+" ["+colors.BOLD+colors.RESULT+"RESULTS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)


def crackseg9k_split():
    import os
    import shutil

    # Base folder
    base_dir = "data/Crackseg9k"

    # Input folders
    images_dir = os.path.join(base_dir, "Images")
    masks_dir = os.path.join(base_dir, "Final_Masks", "Masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("❌ Input directories do not exist. Please check the paths.")
        return
    # Output folders
    output_train_img = os.path.join(base_dir, "train", "images")
    output_train_mask = os.path.join(base_dir, "train", "masks")
    output_test_img = os.path.join(base_dir, "test", "images")
    output_test_mask = os.path.join(base_dir, "test", "masks")

    # Create output directories
    for path in [output_train_img, output_train_mask, output_test_img, output_test_mask]:
        os.makedirs(path, exist_ok=True)

    # Function to copy images/masks according to a txt file
    def copy_files(split):
        txt_path = os.path.join(base_dir, "Final_Masks", f"{split}.txt")
        with open(txt_path, "r") as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        print(f"📦 Processing {split}: {len(filenames)} files")

        for name in filenames:
            # Common pattern: image file might have .jpg/.png extension
            possible_exts = [".png"]
            image_file = None

            for ext in possible_exts:
                candidate = os.path.join(images_dir, name)

                if os.path.exists(candidate):
                    image_file = candidate
                    break

            mask_file = os.path.join(masks_dir, name)  # adjust if mask ext differs

            if image_file and os.path.exists(mask_file):
                if split == "train":
                    shutil.copy2(image_file, output_train_img)
                    shutil.copy2(mask_file, output_train_mask)
                elif split == "test":
                    shutil.copy2(image_file, output_test_img)
                    shutil.copy2(mask_file, output_test_mask)
            else:
                print(f"⚠️ Missing pair for: {name}")

    # Copy train/test sets
    copy_files("train")
    copy_files("test")

    print("✅ Dataset reorganized successfully!")

def plot_filters(img=None):
    """Visualize convolution filters on an image (supports HWC and CHW)."""
    
    # --- Default image ---
    if img is None:
        if hasattr(data, "crack"):
            img = data.crack()
        else:
            img = data.camera()
    
    # --- Convert to numpy array and float32 ---
    if isinstance(img, np.ndarray) is False:
        img = np.array(img)
    img = img.astype(np.float32)

    # --- Handle grayscale or channel-first ---
    if img.ndim == 3:
        if img.shape[0] == 3 and img.shape[-1] != 3:
            # (C, H, W) → (H, W, C)
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        # Expand grayscale to 3 channels for visualization
        img = np.stack([img] * 3, axis=-1)

    # --- Convert to grayscale safely ---
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img[..., 0] if img.ndim == 3 else img

    gray = (gray * 255).astype(np.uint8)

    # --- Define filters ---
    filters = {
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
        "gauss3": cv2.getGaussianKernel(3, 0.8) @ cv2.getGaussianKernel(3, 0.8).T,
    }

    # --- Apply filters ---
    responses = {name: cv2.filter2D(gray.astype(np.float32), -1, k)
                 for name, k in filters.items()}

    # --- Plot results ---
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i, (name, res) in enumerate(responses.items(), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(res, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def ddp_init_if_needed(backend="gloo"):
    """Initialize dist if launched by torchrun; return (rank, world_size)."""
    if dist.is_available() and not dist.is_initialized():
        # torchrun sets these
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
            )
        else:
            rank, world_size = 0, 1
    else:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world_size

def ddp_cleanup_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

# --------- Logging ---------
def rank0_print(*a, **k):
    if not ddp_is_initialized() or dist.get_rank() == 0:
        print(*a, **k)

# --------- Thread hygiene (call early, once) ---------
def tame_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --------- Robust tensor all_gather for 1D vectors ---------
@torch.no_grad()
def all_gather_vector_robust(local_vec: torch.Tensor, pad_fill: float = float("inf")):
    """
    local_vec: 1D CPU tensor (float32/int64 etc.)
    Returns: concatenated list of (rank_idx, unpadded tensor)
    Safe if some ranks have zero length; no Python object pickling.
    """
    assert local_vec.device.type == "cpu" and local_vec.dim() == 1
    if not ddp_is_initialized():
        return [(0, local_vec)]

    world = dist.get_world_size()
    n_local = torch.tensor([local_vec.numel()], dtype=torch.int32)
    size_list = [torch.empty_like(n_local) for _ in range(world)]
    dist.all_gather(size_list, n_local)
    sizes = [int(s.item()) for s in size_list]
    max_local = max(sizes)

    if local_vec.numel() < max_local:
        pad = torch.full((max_local - local_vec.numel(),), pad_fill,
                         dtype=local_vec.dtype)
        local_vec = torch.cat([local_vec, pad], dim=0)

    gathered = [torch.empty_like(local_vec) for _ in range(world)]
    dist.all_gather(gathered, local_vec)

    out = []
    for r in range(world):
        cnt = sizes[r]
        if cnt == 0: 
            out.append((r, local_vec[:0].clone()))
        else:
            out.append((r, gathered[r][:cnt].clone()))
    return out

# --------- Round-robin sharding ---------
def shard_indices(n_items, rank, world):
    return list(range(rank, n_items, world))

def reorganize_dataset(datasetpath, datasetname):
        # ============================================================
    # Configuration
    # ============================================================
    root_dir = datasetpath
    images_dir = os.path.join(root_dir, "images", datasetname)
    masks_dir = os.path.join(root_dir, "masks", datasetname)
    test_file = os.path.join(root_dir, "test.txt")

    test_img_out = os.path.join(root_dir, "test", "images")
    test_mask_out = os.path.join(root_dir, "test", "masks")
    train_img_out = os.path.join(root_dir, "train", "images")
    train_mask_out = os.path.join(root_dir, "train", "masks")

    # ============================================================
    # Create output folders
    # ============================================================
    for d in [test_img_out, test_mask_out, train_img_out, train_mask_out]:
        os.makedirs(d, exist_ok=True)

    # ============================================================
    # Load test file list
    # ============================================================
    test_files = set()
    with open(test_file, "r") as f:
        for line in f:
            img_path, mask_path = line.strip().split()
            img_name = os.path.basename(img_path)
            test_files.add(img_name)

    print(f"[INFO] Loaded {len(test_files)} test files.")

    # ============================================================
    # Split dataset
    # ============================================================
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in all_images:
        mask_name = img_name  # same filename pattern for masks
        mask_name = mask_name.replace(".jpg", ".png")

        img_src = os.path.join(images_dir, img_name)
        mask_src = os.path.join(masks_dir, mask_name)

        if img_name in test_files:
            img_dst = os.path.join(test_img_out, img_name)
            mask_dst = os.path.join(test_mask_out, mask_name)
        else:
            img_dst = os.path.join(train_img_out, img_name)
            mask_dst = os.path.join(train_mask_out, mask_name)

        # copy (you can use move() instead if you prefer)
        shutil.copy2(img_src, img_dst)
        shutil.copy2(mask_src, mask_dst)

    print(f"[DONE] Dataset reorganized:")
    print(f"  → Train images: {len(os.listdir(train_img_out))}")
    print(f"  → Test images:  {len(os.listdir(test_img_out))}")

def devide_crackseg9k(root_in, out_out):
    import os
    import shutil
    from tqdm import tqdm

    # Dataset identifiers (substrings used to detect which dataset each file belongs to)
    DATASET_KEYS = {
        "CRACK500": "CRACK500",
        "DeepCrack": "DeepCrack",
        "GAPS384": "GAPS384",
        "Rissbilder": "Rissbilder",
        "noncrack": "noncrack",
        "cracktree200": "cracktree200",
        "Volker": "Volker",
        "Ceramic": "Ceramic",
        "a_": "AEL",      # or any name you want for 'a_' / 'c_' prefixed sets
        "c_": "CCIC"
    }

    def detect_dataset(name):
        """Detect dataset name based on filename substrings."""
        for key, dataset in DATASET_KEYS.items():
            if key in name:
                return dataset
        return "Unknown"

    def restructure_dataset(root, out):
        for split in ["train", "test"]:
            img_dir = os.path.join(root, split, "images")
            mask_dir = os.path.join(root, split, "masks")

            for img_name in tqdm(os.listdir(img_dir), desc=f"Processing {split} images"):
                dataset_name = detect_dataset(img_name)
                if dataset_name == "Unknown":
                    continue

                # Define output directories
                out_img_dir = os.path.join(out, dataset_name, split, "images")
                out_mask_dir = os.path.join(out, dataset_name, split, "masks")
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_mask_dir, exist_ok=True)

                # Define paths
                src_img = os.path.join(img_dir, img_name)
                src_mask = os.path.join(mask_dir, img_name)
                dst_img = os.path.join(out_img_dir, img_name)
                dst_mask = os.path.join(out_mask_dir, img_name)

                # Copy files (can change to `shutil.move` if you want to move instead)
                shutil.copy2(src_img, dst_img)
                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, dst_mask)
                else:
                    print(f"⚠️ No mask found for {img_name}")
        
    restructure_dataset(root_in, out_out)

if __name__ == "__main__":
    # Root dataset directory (adjust this)
    ROOT = "data/crackseg9k"

    # Target base output folder
    OUT = "data/crackseg9k_split"
    devide_crackseg9k(ROOT, OUT)
    print("\n✅ Dataset restructuring complete!")
