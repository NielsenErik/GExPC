import numpy as np
import random
from abc import ABC, abstractmethod
import copy
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.filters import gabor, sobel, laplace
from skimage.color import rgb2lab, rgb2hsv
from torch import nn

import torch
import torchvision.models as models
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    '''
    In case of pos_weight, it becomes weighted BCE loss.
    '''
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for binary segmentation"""
        pred = torch.sigmoid(pred)   # convert logits → probabilities
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = self.dice_loss(pred, target)
        return 0.5 * bce_loss + 0.5 * d_loss

class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, layer="avgpool"):
        super().__init__()
        backbone = getattr(models, model_name)(pretrained=pretrained)

        # cut at desired layer
        if layer == "avgpool":
            self.encoder = torch.nn.Sequential(*list(backbone.children())[:-2])  # keep conv part
        elif layer == "layer4":
            self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])  # up to last conv block
        else:
            raise ValueError(f"Unsupported layer {layer}")

    def forward(self, x):
        """
        :param x: torch.Tensor (N,3,H,W), values in [0,1]
        :return: torch.Tensor (N,C,Hf,Wf)
        """
        feats = self.encoder(x)  # (N,C,Hf,Wf)
        return feats

# ---------------- Node definitions ----------------
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children is not None else []

    @abstractmethod
    def evaluate(self, x):
        pass
    
class TabularInputNode(Node):
    def __init__(self, feature_idx, distribution):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.distribution = distribution  # torch-based PDF
    def evaluate(self, x):
        return self.distribution(x[:, self.feature_idx])
    
class InputNode(Node):
    def __init__(self, feature_idx, mu=0.0, sigma=1.0):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.mu = mu
        self.sigma = sigma
        self.class_params = {}  # {class_label: (mu, sigma)}

    def fit_supervised(self, X, y):
        """
        Fit per-class Gaussians for this feature.
        :param X: np.ndarray (n_samples, n_features)
        :param y: np.ndarray (n_samples,) class labels
        """
        for c in tqdm.tqdm(np.unique(y), desc=f"Fitting InputNode feature {self.feature_idx}"):
            vals = X[y == c, self.feature_idx]
            mu= np.median(vals)
            sigma = np.mean(np.abs(vals - mu)) + 1e-6

            self.class_params[c] = (mu, sigma)

    def evaluate_class(self, X, class_idx):
        """Evaluate likelihood given class-conditional params."""
        vals = X[:, self.feature_idx]
        mu, sigma = self.class_params[class_idx]
        return norm.pdf(vals, loc=mu, scale=sigma)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor (B,C,H,W) or (B,n_features)
        Returns: torch.Tensor (B,H,W) or (B,)
        """
        if isinstance(X, np.ndarray):  # safety: convert if np accidentally passed
            X = torch.from_numpy(X).float()

        vals = X[:, self.feature_idx]  # (B,) if tabular, or (B,H,W) if per-pixel
        mu = torch.tensor(self.mu, dtype=torch.float32, device=X.device)
        sigma = torch.tensor(self.sigma, dtype=torch.float32, device=X.device)
        probs = torch.exp(-0.5 * ((vals - mu) / (sigma + 1e-12))**2) / (sigma * (2*np.pi)**0.5)
        return probs

    def set_params(self, mu, sigma):
        """Update distribution parameters."""
        self.mu = mu
        self.sigma = sigma

class SumNode(Node):
    def __init__(self, children, weights=None, device="cpu"):
        super().__init__(children)
        self.n_children = len(children)
        if weights is None:
            init_w = torch.ones(self.n_children, device=device) / self.n_children
        else:
            if isinstance(weights, torch.Tensor):
                init_w = weights.detach().clone().float().to(device)
            else:
                init_w = torch.tensor(weights, dtype=torch.float32, device=device)
            init_w = init_w / init_w.sum()
        # make weights a Parameter so optimizer can update them
        self.weights = nn.Parameter(init_w.clone().float(), requires_grad=True)

    def set_weights(self, weights):
        assert len(weights) == self.n_children, "Weights length must match number of children"
        with torch.no_grad():
            w = torch.tensor(weights, dtype=torch.float32, device=self.weights.device)
            w = w / w.sum()
            self.weights.copy_(w)

    def evaluate(self, x):
        # child.evaluate(x) must return torch.Tensor shaped (B, H, W)
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        # tensordot weights (n_children,) with child_vals -> (B, H, W)
        weighted_sum = torch.tensordot(self.weights, child_vals, dims=([0], [0]))  # (B, H, W)
        return weighted_sum

class ProductNode(Node):
    def __init__(self, children):
        super().__init__(children)

    def evaluate(self, x):
        child_vals = torch.stack([child.evaluate(x) for child in self.children], dim=0)  # (n_children, B, H, W)
        return torch.prod(child_vals, dim=0)  # (B, H, W)
    
class ClassifierNode(Node):
    """
    Stacks its children outputs as logits across the class dimension.
    Each child should evaluate to (B, H, W) logits/scores for that class.
    Returns: logits tensor (B, n_classes, H, W)
    """
    def __init__(self, children):
        super().__init__(children)
        # no internal weights here — children provide class scores

    def evaluate(self, x):
        # child.evaluate(x) -> (B, H, W)
        child_vals = [child.evaluate(x) for child in self.children]  # list of (B, H, W)
        logits = torch.stack(child_vals, dim=1)  # (B, n_classes, H, W)
        return logits

# ---------------- Abstract Graph ----------------
class AbstractGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # parent -> set(children)

    def add_node(self, node):        
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must be added before creating an edge")
        self.edges[parent].add(child)

    def get_children(self, node):
        return self.edges.get(node, set())

    def get_parents(self, node):
        return {p for p, children in self.edges.items() if node in children}

# ---------------- DAG (Acyclic Graph) ----------------
class DirectedAcyclicGraph(AbstractGraph):
    def __init__(self):
        super().__init__()

    def add_edge(self, parent, child):
        super().add_edge(parent, child)
        if self._has_cycle():
            self.edges[parent].remove(child)
            raise ValueError("Adding this edge creates a cycle")

    def _has_cycle(self):
        visited = set()
        stack = set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in self.get_children(node):
                if visit(child):
                    return True
            stack.remove(node)
            return False

        return any(visit(node) for node in self.nodes)

# Probabilistic Circuit
class ProbabilisticCircuit:
    """
    Probabilistic Circuit for image segmentation.
    Evaluates pixel-wise likelihoods for class labels.
    """
    def __init__(self, n_classes=2, device="cpu"):
        self.graph = DirectedAcyclicGraph()
        self.root = None
        self.n_classes = n_classes
        self.device = device

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, parent, child):
        self.graph.add_edge(parent, child)
        if child not in parent.children:
            parent.children.append(child)

    def set_root(self, node):
        if node not in self.graph.nodes:
            raise ValueError("Root must be in the graph")
        self.root = node

    def evaluate(self, input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the segmentation PC.
        image_tensor: torch.Tensor of shape (B, C, H, W)
        Returns: torch.Tensor of shape (B, n_classes, H, W)
        """
        if self.root is None:
            raise ValueError("Root not set")
        return self.root.evaluate(input)
    
    def predict_mask(self, image_tensor):
        """
        Returns segmentation mask from PC evaluation.
        """
        probs = self.evaluate(image_tensor)  # (B, n_classes, H, W)
        return torch.argmax(probs, dim=2)    # (B, H, W)
    
    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, children in self.graph.edges.items() for c in children]

    def visualize(self):
        """
        Visualize the circuit using networkx.
        """
        G = nx.DiGraph()
        for node in self.get_nodes():
            label = node.__class__.__name__
            if isinstance(node, InputNode):
                label += f"\n(feature={node.feature_idx})"
            elif isinstance(node, SumNode):
                weights_str = ','.join([f"{w:.2f}" for w in node.weights.detach().cpu().numpy()])
                label += f"\n(w={weights_str})"
            G.add_node(node, label=label)

        for parent, child in self.get_edges():
            G.add_edge(parent, child)

        pos = nx.kamada_kawai_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=300, node_color="lightblue")
        plt.show()

class PCNet(ProbabilisticCircuit):
    def __init__(self,n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3, feature_extractor="handcrafted"):
        
        super().__init__(n_classes, device)
        self.network = None  # Placeholder for a pc network
        self.max_depth = None
        self.max_branching = None
        self.distribution = distribution
        self.n_classes = n_classes
        self.device = device
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.feature_tensor = None
        self.feature_extractor = feature_extractor

    def pairwise(self, nodes):
        """Group nodes pairwise for ProductNode layer"""
        random.shuffle(nodes)
        paired = []
        for i in range(0, len(nodes)-1, 2):
            paired.append((nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            paired.append((nodes[-1],))
        return paired
    
    def branchwise(self, nodes):
        """Group nodes randomly with max_branching for SumNode layer"""
        random.shuffle(nodes)
        branched = []
        current = nodes[:]
        while len(current) > self.max_branching:
            group_size = random.randint(1, self.max_branching)
            branched.append(tuple(current[:group_size]))
            current = current[group_size:]
        if len(current) > 0:
            branched.append(tuple(current))
        return branched
    
    def prod_layer(self, current_nodes):
        """Build a ProductNode layer"""
        next_nodes = []
        pairs = self.pairwise(current_nodes)
        for pair in pairs:
            if len(pair) == 1:
                next_nodes.append(pair[0])
                continue
            prod_node = ProductNode(children=list(pair))
            self.add_node(prod_node)
            for child in pair:
                self.add_edge(prod_node, child)
            next_nodes.append(prod_node)
        return next_nodes
    
    def sum_layer(self, current_nodes):
        """Build a SumNode layer"""
        next_nodes = []
        branches = self.branchwise(current_nodes)
        for branch in branches:
            # Random weights
            weights = torch.rand(len(branch), device=self.device)

            # Normalize so sum = 1
            weights = weights / weights.sum()
            sum_node = SumNode(children=list(branch), weights=weights, device=self.device)
            self.add_node(sum_node)
            for child in branch:
                self.add_node(child)
                self.add_edge(sum_node, child)
            next_nodes.append(sum_node)
        return next_nodes
    
    def init_network(self, current_nodes):
        """Iteratively build the tree layers instead of recursively."""
        nodes = current_nodes
        for depth in trange(1, self.max_depth + 1, desc="Building PCNet"):
            if depth == self.max_depth:
                # Last layer: create root ClassifierNode
                root = ClassifierNode(nodes)
                self.add_node(root)
                for child in nodes:
                    self.add_edge(root, child)
                return root
            else:
                if depth % 2 == 0:
                    nodes = self.prod_layer(nodes)
                else:
                    nodes = self.sum_layer(nodes)
        return root  # fallback, though loop returns at max_depth
    
    def get_handcrafted_features(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Compute handcrafted feature bank for a list of images.

        :param images: list of np.ndarray, each (H,W,3) RGB in [0,255] or [0,1]
        :return: feature_tensor: np.ndarray (N,C,H,W)
                N = number of images
                C = number of feature channels
                H,W = spatial resolution
        """
        
        def process_one(image: np.ndarray) -> np.ndarray:
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0  # normalize to [0,1]
            C, H, W = image.shape
            assert C == 3, f"Expected RGB image with 3 channels, got shape {image.shape}"

            features = []

            # --- 1. Raw color channels ---
            for c in range(3):
                features.append(image[c, ...])  # shape (H,W)

            # Convert to (H,W,3) for color transforms
            image_hw3 = np.transpose(image, (1, 2, 0))  # (H,W,3)

            # --- 2. Color transforms ---
            lab = rgb2lab(image_hw3)  # (H,W,3)
            hsv = rgb2hsv(image_hw3)  # (H,W,3)
            for c in range(3):
                features.append(lab[..., c])
                features.append(hsv[..., c])

            # --- 3. Edge / texture ---
            gray = cv2.cvtColor((image_hw3*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
            features.append(sobel(gray))
            features.append(laplace(gray))

            # --- 4. Gabor filters ---
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                filt_real, filt_imag = gabor(gray, frequency=0.2, theta=theta)
                features.append(filt_real)
                features.append(filt_imag)

            # Stack into (C,H,W)
            return np.stack(features, axis=0).astype(np.float32)

        # Process each image
        feature_tensors = [process_one(img) for img in tqdm.tqdm(images, desc="Computing handcrafted features")]

        # Stack into (N,C,H,W)
        return np.stack(feature_tensors, axis=0)

    def get_cnn_features(self, input_images: list[np.ndarray]) -> np.ndarray:
        """
        Compute CNN feature embeddings for a list of images.
        
        :param input_images: list of np.ndarray, each (H,W,3) RGB in [0,255] or [0,1]
        :return: feature_tensor: np.ndarray (N,C,Hf,Wf)
        """
        self.cnn.eval()
        device = next(self.cnn.parameters()).device

        tensors = []
        for img in input_images:
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            if img.shape[-1] == 3:  # (H,W,3) → (3,H,W)
                img = np.transpose(img, (2,0,1))
            t = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,3,H,W)
            tensors.append(t)

        batch = torch.cat(tensors, dim=0)  # (N,3,H,W)
        with torch.no_grad():
            feats = self.cnn(batch)  # (N,C,Hf,Wf)

        return feats.cpu().numpy().astype(np.float32)

    def get_feature_tensor(self, input_images):
        if self.feature_extractor == "handcrafted":
            return self.get_handcrafted_features(input_images)
        elif self.feature_extractor == "cnn":
            self.cnn = CNNFeatureExtractor(model_name="resnet18", pretrained=True, layer="layer4").to(self.device)
            return self.get_cnn_features(input_images)
        else:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
    
    def set_input_nodes(self, input_images, labels):
        """
        Build supervised InputNodes using image features and GT labels.
        input_images: np.ndarray (B,H,W,3)
        labels: np.ndarray (B,H,W) ground truth segmentation masks
        """
        self.feature_tensor = self.get_feature_tensor(input_images)  # (B,C,H,W)
        B, C, H, W = self.feature_tensor.shape

        # Flatten
        features = self.feature_tensor.transpose(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
        labels = labels.reshape(-1)                                      # (B*H*W,)

        # Create input nodes for each feature dim
        input_nodes = []
        for i in range(C):
            node = InputNode(i)
            node.fit_supervised(features, labels)
            input_nodes.append(node)
            self.add_node(node)

        return input_nodes

    def fit_supervised(self, images, masks, epochs=5, lr=0.01):
        # Step 1: Build input nodes with supervision
        input_nodes = self.set_input_nodes(images, masks)

        # Step 2: Build circuit structure
        print("Building circuit...")
        self.root = self.init_network(input_nodes)
        self.set_root(self.root)
        print("Circuit built.")
        # Collect trainable parameters: all SumNode weights and any nn.Parameter in other nodes
        params = []
        for node in tqdm.tqdm(self.graph.nodes, desc="Collecting parameters"):
            # if node has attribute 'weights' and it's a nn.Parameter, include it
            if hasattr(node, 'weights') and isinstance(node.weights, torch.nn.Parameter):
                params.append(node.weights)

        optimizer = torch.optim.Adam(params, lr=lr)

        if self.feature_tensor is None:
            self.feature_tensor = torch.from_numpy(
                self.get_feature_tensor(images)
            ).float().to(self.device)
        target = torch.from_numpy(masks).long().to(self.device)  # (B,1,H,W) maybe

        # ensure target shape is (B,H,W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        pbar = tqdm.tqdm(range(epochs), desc="Training PC", ncols=100)
        for epoch in pbar:
            optimizer.zero_grad()
            logits = self.evaluate(self.feature_tensor)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            
            # Update progress bar postfix with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    def evaluate(self, data):
        return self.root.evaluate(data)
    
    def predict_mask(self, image):
        """
        images: numpy list or torch tensor already processed to shape (B,C,H,W)
        returns: numpy array (B,H,W) with predicted class ids
        """
        if isinstance(image, list) or isinstance(image, np.ndarray):
            # if list of numpy images: compute handcrafted features first
            if isinstance(image, list):
                feature_tensor = torch.from_numpy(self.get_feature_tensor(image)).float().to(self.device)
            else:
                # if user passed (B,C,H,W) numpy directly:
                feature_tensor = torch.from_numpy(image).float().to(self.device)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            feature_tensor = self.get_handcrafted_features([image])
        else:
            raise ValueError("Unsupported input type for predict_mask")

        with torch.no_grad():
            logits = self.evaluate(feature_tensor)            # (B, n_classes, H, W)
            preds = torch.softmax(logits, dim=2)               # (B, H, W)
            return preds.cpu().numpy()



if __name__ == "__main__":
    pc_net = PCNet(input_size=(1, 28, 28), patch_size=(9,9), n_classes=2, distribution=None, device="cpu", max_depth=4, max_branching=3)

    # Dummy data
    B, C, H, W = 10, 1, 28, 28
    image = torch.rand(B, C, H, W)
    pc_net.initialize_network(image)
    probs = pc_net.evaluate(image)
    mask = pc_net.predict_mask(image)
    probs = probs.detach().cpu().numpy()
    print(mask.shape, probs.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(image[0,0].numpy(), cmap='gray')
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask[0].numpy(), cmap='gray')
    plt.title("Predicted Segmentation Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(probs[0,0], cmap='hot')
    plt.title("Predicted Probabilities")
    plt.show()

    pc_net.visualize()