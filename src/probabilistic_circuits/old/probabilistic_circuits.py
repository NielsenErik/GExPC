import numpy as np
import random
from abc import ABC, abstractmethod
import copy
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import torch

# ---------------- Node definitions ----------------
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children is not None else []

    @abstractmethod
    def evaluate(self, x):
        pass

class InputNode(Node):
    def __init__(self, feature_idx, distribution):
        super().__init__(children=[])
        self.feature_idx = feature_idx
        self.distribution = distribution  # callable: p(x)

    def evaluate(self, x):
        return self.distribution(x[:, self.feature_idx])

class SumNode(Node):
    '''
    The SumNode
    class represents a node in a probabilistic circuit that computes a weighted sum of its child nodes' evaluations.

    Here's what each class method does:

    init(self, children, weights=None)
    : Initializes a 
    SumNode
    with a list of child nodes and optional weights. If no weights are provided, it defaults to equal weights for all children.
    evaluate(self, x)
    : Evaluates the weighted sum of the child nodes' evaluations for a given input x.
    set_weights(self, weights)
    : Sets new weights for the child nodes, ensuring that the weights sum to 1.
    '''
    def __init__(self, children, weights=None):
        super().__init__(children)
        self.n_children = len(children)
        if weights is None:
            weights = np.ones(self.n_children) / self.n_children
        self.weights = np.array(weights, dtype=float)
        self.weights /= np.sum(self.weights)

    def evaluate(self, x):
        child_vals = np.array([child.evaluate(x) for child in self.children])
        return np.dot(self.weights, child_vals)

    def set_weights(self, weights):
        assert len(weights) == self.n_children, "Weights length must match number of children"
        self.weights = np.array(weights, dtype=float)
        self.weights /= np.sum(self.weights)

class ProductNode(Node):
    '''
    The 
    ProductNode
    class represents a node in a probabilistic circuit that computes the product of its child nodes' evaluations.

    Here's what each method does:

    init(self, children)
    :params children (list of Node)
    : Initializes a 
    ProductNode
    with a list of child nodes.
    evaluate(self, x)
    : Evaluates the product of the child nodes' evaluations for a given input x, returning the result as a numpy array.
    '''
    def __init__(self, children):
        super().__init__(children)

    def evaluate(self, x):
        child_vals = np.array([child.evaluate(x) for child in self.children])
        return np.prod(child_vals, axis=0)

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
class ProbabilisticCircuit(DirectedAcyclicGraph):
    """
    A Probabilistic Circuit built on a Directed Acyclic Graph (DAG).
    - Nodes: InputNode, SumNode, ProductNode
    - Supports EM learning of SumNode weights
    - Can evaluate inputs and apply to images for segmentation
    """
    def __init__(self, root: Node):
        super().__init__()
        self.root = root
        self.add_node(root)

    # ---------------- Inference ----------------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit likelihood for given inputs.
        :param x: (n_samples, n_features)
        :return: likelihoods (n_samples,)
        """
        return self.root.evaluate(x)

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Return log-likelihood to avoid underflow."""
        return np.log(self.evaluate(x) + 1e-12)

    # ---------------- EM Training ----------------
    def e_step(self, x: np.ndarray):
        """
        Expectation step: Compute responsibilities for sum nodes.
        :param x: (n_samples, n_features)
        :return: dict mapping SumNode -> responsibilities
        """
        responsibilities = {}
        for node in self.nodes:
            if isinstance(node, SumNode):
                child_vals = np.array([child.evaluate(x) for child in node.children])  # (n_children, n_samples)
                weighted = node.weights[:, None] * child_vals
                norm = np.sum(weighted, axis=0, keepdims=True) + 1e-12
                resp = weighted / norm  # shape (n_children, n_samples)
                responsibilities[node] = resp
        return responsibilities

    def m_step(self, responsibilities: dict):
        """
        Maximization step: Update weights of SumNodes.
        """
        for node, resp in responsibilities.items():
            # Average responsibilities across samples
            new_weights = resp.mean(axis=1)
            node.set_weights(new_weights)

    def fit(self, x: np.ndarray, n_iters=10):
        """
        EM algorithm to learn SumNode weights.
        """
        for _ in range(n_iters):
            resp = self.e_step(x)
            self.m_step(resp)

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class PCNet(ProbabilisticCircuit):
    """
    Deep Tree-Structured Probabilistic Circuit (PCNet) for supervised segmentation.
    Alternates Product/Sum layers and allows tree visualization.
    """
    def __init__(self, input_nodes: list, input_size: tuple , max_depth: int = 3, max_branching: int = 2):
        """
        :param input_nodes: list of InputNodes (leaf distributions)
        :param input_size: tuple, e.g., (H, W, C) for image inputs
        :param max_depth: number of alternating Product/Sum layers
        :param max_branching: max number of children per SumNode
        """
        self.input_nodes = input_nodes
        self.input_size = input_size
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.nodes = set()
        self.edges = {}

        # Build tree recursively
        print("Building PCNet structure...")
        self.root = self.set_next_layer(self.input_nodes, 1)
        super().__init__(self.root)

    # ---------------- Layer building helpers ----------------
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
            sum_node = SumNode(children=list(branch))
            self.add_node(sum_node)
            for child in branch:
                self.add_node(child)
                self.add_edge(sum_node, child)
            next_nodes.append(sum_node)
        return next_nodes
    
    def set_next_layer(self, current_nodes, depth):
        """Recursively build the tree layers"""
        if depth == self.max_depth:
            # If multiple nodes remain, create a root SumNode
            if len(current_nodes) == 1:
                return current_nodes[0]
            else:
                root = SumNode(current_nodes)
                self.add_node(root)
                for child in current_nodes:
                    self.add_edge(root, child)
                return root
        else:
            if depth % 2 == 0:
                next_nodes = self.prod_layer(current_nodes)
            else:
                next_nodes = self.sum_layer(current_nodes)
            return self.set_next_layer(next_nodes, depth+1)

    # ---------------- Evaluation ----------------
    def evaluate(self, x):
        """Evaluate circuit likelihood for input x"""
        return super().evaluate(x)

    # ---------------- Tree utilities ----------------
    def print_tree(self, node=None, level=0):
        """Print indented tree structure"""
        if node is None:
            node = self.root
        indent = "  " * level
        node_type = type(node).__name__
        print(f"{indent}{node_type} (id={id(node)})")
        for child in node.children:
            self.print_tree(child, level + 1)

    def export_tree_dict(self, node=None):
        """Export tree as nested dictionary"""
        if node is None:
            node = self.root
        return {f"{type(node).__name__}_{id(node)}": [self.export_tree_dict(c) for c in node.children]}

    def visualize_tree(self, node=None):
        """Visualize tree using networkx + matplotlib"""
        if node is None:
            node = self.root

        G = nx.DiGraph()

        def add_edges(n):
            for child in n.children:
                G.add_edge(f"{type(n).__name__}_{id(n)}", f"{type(child).__name__}_{id(child)}")
                add_edges(child)

        add_edges(node)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=8, arrowsize=20)
        plt.show()

    def _eval_node_for_class(self, node, x, cls):
        """
        Evaluate `node` on input x conditioning InputNodes on class `cls`.
        :param node: Node (InputNode / SumNode / ProductNode)
        :param x: np.ndarray shape (N, Cfeatures)
        :param cls: class id (int)
        :return: np.ndarray length N of likelihoods p(node | x, class=cls)
        """
        if isinstance(node, InputNode):
            # If class params exist, use them; otherwise fallback to node.distribution
            params = getattr(node, "class_params", None)
            if params is not None and cls in params:
                mu, sigma = params[cls]
                # safe pdf
                return norm.pdf(x[:, node.feature_idx], loc=mu, scale=sigma + 1e-12)
            else:
                # fallback: node.distribution expects x with shape (N, features)
                return node.evaluate(x)
        elif isinstance(node, ProductNode):
            # product of children (vectorized)
            child_vals = np.array([self._eval_node_for_class(child, x, cls) for child in node.children])
            return np.prod(child_vals, axis=0)
        elif isinstance(node, SumNode):
            # weighted sum of children (each child conditioned on cls)
            child_vals = np.array([self._eval_node_for_class(child, x, cls) for child in node.children])  # (n_children, N)
            return np.dot(node.weights, child_vals)
        else:
            raise TypeError("Unknown node type in _eval_node_for_class")

    # ---------- Helper: set class-conditional params for InputNodes ----------
    def _estimate_leaf_class_params(self, X, y, classes):
        """
        Estimate Gaussian mean and std for each class and input feature.
        Stores in self.class_params: dict[class -> {feature_idx: (mu, sigma)}]
        """
        self.class_params = {}
        for cls in classes:
            cls_dict = {}
            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue
            for node in self.input_nodes:
                vals = X_cls[:, node.feature_idx]
                mu, sigma = np.mean(vals), np.std(vals) + 1e-6
                cls_dict[node.feature_idx] = (mu, sigma)
                # Update node distribution too
                node.distribution = lambda v, mu=mu, sigma=sigma: (
                    1.0 / (np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((v - mu)/sigma)**2)
                )
            self.class_params[cls] = cls_dict

    # ---------- Supervised training ----------
    def fit_supervised(self, X_list, y_list, epochs: int = 5, verbose: bool = True):
        """
        Supervised training using multiple images + ground-truth masks.

        :param X_list: list of images, each of shape (C,H,W) or (H,W,C)
        :param y_list: list of masks, each of shape (1,H,W) or (H,W)
        :param n_iters: number of EM-style iterations
        :param verbose: print progress
        """
        print("Starting supervised training of PCNet...")
        all_X = []
        all_y = []

        for X, y in zip(X_list, y_list):
            # ---- Handle channel-first images (C,H,W) ----
            if X.ndim == 3 and X.shape[0] <= 4:  # likely (C,H,W)
                X = np.transpose(X, (1, 2, 0))   # -> (H,W,C)

            if y.ndim == 3 and y.shape[0] == 1:  # (1,H,W)
                y = y[0]                         # -> (H,W)

            if X.ndim != 3 or y.ndim != 2:
                raise ValueError(f"Invalid shapes: got X={X.shape}, y={y.shape}")
            H, W, C = X.shape
            all_X.append(X.reshape(-1, C))  # (H*W, C)
            all_y.append(y.reshape(-1))     # (H*W,)

        # Concatenate across all images
        X_flat = np.concatenate(all_X, axis=0)   # (N_total, C)
        y_flat = np.concatenate(all_y, axis=0)   # (N_total,)
        classes = np.unique(y_flat)

        # --------- 1. Initialize leaf params ---------
        self._estimate_leaf_class_params(X_flat, y_flat, classes=classes)

        # --------- 2. EM iterations ---------
        for it in trange(epochs):
            if verbose:
                print(f"[PCNet supervised training] Iter {it+1}/{epochs}")

            responsibilities = {}
            indices_by_class = {cls: np.where(y_flat == cls)[0] for cls in classes}

            # ---- E-step: responsibilities ----
            for node in list(self.nodes):
                if not isinstance(node, SumNode):
                    continue

                n_children = len(node.children)
                N = X_flat.shape[0]
                child_vals = np.zeros((n_children, N), dtype=float)

                for cls in classes:
                    idxs = indices_by_class.get(cls, np.array([], dtype=int))
                    if idxs.size == 0:
                        continue
                    X_cls = X_flat[idxs]
                    for ci, child in enumerate(node.children):
                        child_vals[ci, idxs] = self._eval_node_for_class(child, X_cls, cls)

                weighted = node.weights[:, None] * child_vals
                denom = np.sum(weighted, axis=0, keepdims=True) + 1e-12
                resp = weighted / denom
                responsibilities[node] = resp

            # ---- M-step: update weights ----
            for node, resp in responsibilities.items():
                new_weights = resp.mean(axis=1)
                new_weights = new_weights + 1e-8
                node.set_weights(new_weights)

            # ---- Re-estimate leaf params ----
            self._estimate_leaf_class_params(X_flat, y_flat, classes=classes)


        if verbose:
            print("✅ PCNet supervised training finished.")

    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation mask for a single image.
        :param image: np.ndarray of shape (C,H,W) or (H,W,C)
        :return: mask of shape (H,W) with predicted class labels
        """
        # ---- Handle channel-first input (C,H,W) ----
        if image.ndim == 3 and image.shape[0] <= 4:  # C,H,W
            image = np.transpose(image, (1, 2, 0))  # -> H,W,C

        H, W, C = image.shape
        X_flat = image.reshape(-1, C)  # (H*W, C)

        classes = sorted(self.class_params.keys())
        log_probs = np.zeros((len(classes), X_flat.shape[0]), dtype=float)

        for ci, cls in enumerate(classes):
            cls_vals = []
            for node in self.input_nodes:
                mu, sigma = self.class_params[cls][node.feature_idx]
                # Explicit Gaussian log-likelihood (no lambda call)
                cls_vals.append( -0.5 * np.log(2 * np.pi * sigma**2)- 0.5 * ((X_flat[:, node.feature_idx] - mu) / sigma) ** 2)
            
            log_probs[ci] = np.sum(cls_vals, axis=0)  # sum log-likelihoods across features

        preds = np.argmax(log_probs, axis=0)  # (H*W,)
        return preds.reshape(H, W)
    

""""
from scipy.stats import norm

H, W, C = 4, 4, 3  # toy RGB image

# InputNodes = distributions for each pixel-channel
input_nodes = []
for i in range(H*W*C):
    mu = np.random.randint(50, 200)
    sigma = 20
    input_nodes.append(InputNode(i % C, lambda v, mu=mu, sigma=sigma: norm.pdf(v, loc=mu, scale=sigma)))


# Build tree-structured circuit
tree_pc = TreeProbCircuit(
    input_nodes=input_nodes,
    image_shape=(H, W, C),
    patch_size=(2,2),
    depth=3,
    n_sums=2
)

# Evaluate on one pixel vector
x = np.random.randint(0, 255, size=(1, C))
print("Likelihood:", tree_pc.evaluate(x))
"""