# ============================================================
# Imports
# ============================================================
import random
from abc import ABC, abstractmethod
import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.stats import norm
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import gabor, laplace, sobel
from torch import nn
from tqdm import trange
from collections import Counter

from utils import print_configs, print_debugging, print_info


# ============================================================
# Node Classes
# ============================================================
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children else []

    @abstractmethod
    def evaluate(self, x):
        pass


# ======================================================
# Input Node: returns log-probability
# ======================================================
class InputNode(nn.Module, Node):
    """
    Input node that automatically learns the best-fitting distribution
    among {Gaussian, Laplace, Student-t}.
    """
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=[])

        self.feature_idx = feature_idx
        self.device = device

        # Shared location and scale
        self.mu = nn.Parameter(torch.tensor(float(mu_init), dtype=torch.float32, device=device))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init), dtype=torch.float32, device=device)))

        # Optional shape parameter for Student-t
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0, dtype=torch.float32, device=device)))  # degrees of freedom

        # Mixture weights over distributions: [Gaussian, Laplace, Student-t]
        self.logits = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

        # Mixture gate
        self.gate = nn.Parameter(torch.tensor(1.0, device=device))  # learnable

        # Feature name
        self.feature_name = f"feat_{feature_idx}"

    def get_learnable_params(self):
        return [self.mu, self.log_sigma, self.log_nu, self.logits, self.gate]

    def _log_gaussian(self, x, mu, sigma):
        return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma * (2 * np.pi)**0.5)

    def _log_laplace(self, x, mu, sigma):
        b = sigma / np.sqrt(2.0)
        return -torch.abs(x - mu) / b - torch.log(2 * b)

    def _log_student(self, x, mu, sigma, nu):
        z = (x - mu) / (sigma + 1e-12)
        return (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * np.pi)
            - torch.log(sigma + 1e-12)
            - ((nu + 1) / 2) * torch.log1p((z ** 2) / nu)
        )
    
    def fit(self, X):
        """
        Initialize the distribution parameters (mu, sigma, nu, mixture weights)
        based on the dataset statistics for this feature channel.

        Args:
            X (torch.Tensor or np.ndarray): shape (B, C, H, W)
        """
        # Convert to NumPy for easy aggregation
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        # Select this feature channel
        vals = X[:, self.feature_idx].reshape(-1)

        # Robust initialization: median + MAD
        mu_init = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu_init)) + 1e-6)
        sigma_init = mad * 1.4826  # consistent with Gaussian std

        # Use moderate ν to start (Student-t)
        nu_init = 5.0

        # Start with uniform mixture over {Gaussian, Laplace, Student-t}
        logits_init = np.zeros(3, dtype=np.float32)

        # Copy into parameters without breaking the computational graph
        with torch.no_grad():
            self.mu.copy_(torch.tensor(mu_init, dtype=torch.float32, device=self.device))
            self.log_sigma.copy_(torch.log(torch.tensor(sigma_init, dtype=torch.float32, device=self.device)))
            self.log_nu.copy_(torch.log(torch.tensor(nu_init, dtype=torch.float32, device=self.device)))
            self.logits.copy_(torch.tensor(logits_init, dtype=torch.float32, device=self.device))


    def evaluate(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        vals = X[:, self.feature_idx]
        sigma = F.softplus(self.log_sigma) + 1e-6
        nu    = F.softplus(self.log_nu) + 1e-3      
        weights = torch.softmax(self.logits, dim=0)

        log_gauss = self._log_gaussian(vals, self.mu, sigma)
        log_lapl  = self._log_laplace(vals, self.mu, sigma)
        log_stud  = self._log_student(vals, self.mu, sigma, nu)

        # log-sum-exp over distributions
        logpdfs = torch.stack([log_gauss, log_lapl, log_stud], dim=0)
        log_mix = torch.logsumexp(torch.log(weights).view(-1, *([1]*(logpdfs.ndim-1))) + logpdfs, dim=0)
        return log_mix * self.gate # scale by gate


# ======================================================
# Gate Node: learnable mixture of two children
# ======================================================
class GateNode(Node, nn.Module):
    def __init__(self, a, b, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, [a, b])
        self.alpha = nn.Parameter(torch.tensor(0.0, device=device))
        self.device = device

    def get_learnable_params(self):
        return [self.alpha]

    def evaluate(self, x):
        gate = torch.sigmoid(self.alpha)
        left, right = self.children
        llog = left.evaluate(x)
        rlog = right.evaluate(x)
        return torch.logaddexp(llog + torch.log(gate + 1e-8),
                               rlog + torch.log(1 - gate + 1e-8))


# ======================================================
# Residual Node: blend of base and subcircuit
# ======================================================
class ResidualNode(Node, nn.Module):
    def __init__(self, base, sub, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, [base, sub])
        self.beta = nn.Parameter(torch.tensor(0.5, device=device))
        self.device = device

    def get_learnable_params(self):
        return [self.beta]

    def evaluate(self, x):
        b, s = self.children[0].evaluate(x), self.children[1].evaluate(x)
        mix = torch.sigmoid(self.beta)
        return torch.logsumexp(torch.stack([
            b + torch.log(1 - mix + 1e-8),
            s + torch.log(mix + 1e-8)
        ]), dim=0)

# ======================================================
# Sum Node: log-sum-exp over children
# ======================================================
class SumNode(Node, nn.Module):
    def __init__(self, children, weights=None, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children)
        n = len(children)
        if weights is None:
            weights = torch.ones(n, device=device) / n
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        # store log-weights (trainable)
        log_w = torch.log(weights / weights.sum())
        self.weights = nn.Parameter(log_w, requires_grad=True)
        self.device = device

    def get_learnable_params(self):
        return [self.weights]

    def evaluate(self, x):
        """Compute log(∑_i w_i * p_i) = logsumexp(log w_i + log p_i)."""
        child_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)
        log_w = torch.log_softmax(self.weights, dim=0).view(-1, *([1]*(child_logs.ndim-1)))
        return torch.logsumexp(log_w + child_logs, dim=0)


# ======================================================
# Product Node: sum of log-probabilities
# ======================================================
class ProductNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)

    def evaluate(self, x):
        """Compute log(∏_i p_i) = ∑_i log p_i."""
        child_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)
        return torch.sum(child_logs, dim=0)


# ======================================================
# Classifier Node: stack log-likelihoods as class logits
# ======================================================
class ClassifierNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)

    def evaluate(self, x):
        """Return (B, n_classes, H, W) log-probs."""
        child_logs = [c.evaluate(x) for c in self.children]
        return torch.stack(child_logs, dim=1)




# ============================================================
# Graph Classes
# ============================================================
class AbstractGraph:
    def __init__(self):
        self.nodes, self.edges = set(), {}

    def add_node(self, node):
        self.nodes.add(node)
        self.edges.setdefault(node, set())

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must be added before edge")
        self.edges[parent].add(child)


class DirectedAcyclicGraph(AbstractGraph):
    def add_edge(self, parent, child):
        super().add_edge(parent, child)
        if self._has_cycle():
            self.edges[parent].remove(child)
            raise ValueError("Adding this edge creates a cycle")

    def _has_cycle(self):
        visited, stack = set(), set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            if any(visit(c) for c in self.edges.get(node, [])):
                return True
            stack.remove(node)
            return False

        return any(visit(n) for n in self.nodes)


# ============================================================
# Probabilistic Circuit
# ============================================================
class ProbabilisticCircuit:
    def __init__(self, n_classes=2, device="cpu"):
        self.graph = DirectedAcyclicGraph()
        self.root, self.n_classes, self.device = None, n_classes, device

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

    def evaluate(self, x):
        return self.root.evaluate(x) if self.root else None

    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, cs in self.graph.edges.items() for c in cs]

    # visualization function unchanged ...


# ============================================================
# PCNet and EvoPCNet (keep as-is but cleaned)
# ============================================================
# [Your PCNet implementation cleaned goes here ... same structure but formatted]


    def visualize(self, direction="LR", layer_sep=2.0, node_sep=5.0, save_path=None):
        """
        Visualize the probabilistic circuit as a tree-like hierarchical layout with proper spacing.

        :param direction: 'TB' (top-to-bottom) or 'LR' (left-to-right)
        :param layer_sep: spacing between layers (higher = more vertical spacing)
        :param node_sep: spacing between nodes in the same layer (higher = more horizontal spacing)
        """
        G = nx.DiGraph()
        colors = {}

        # Add nodes with labels and colors
        for node in self.get_nodes():
            if isinstance(node, InputNode):
                label = "In"
                label += f"\n{node.feature_name}"
                colors[node] = "lightgreen"
            elif isinstance(node, SumNode):
                label = "Sum"
                weights_str = ','.join([f"{w:.2f}\n" for w in node.weights.detach().cpu().numpy()])
                label += f"\n(w={weights_str})"
                colors[node] = "skyblue"
            elif isinstance(node, ProductNode):
                label = "Product"
                colors[node] = "lightcoral"
            elif isinstance(node, ClassifierNode):
                label = "Classifier"
                colors[node] = "orange"
            else:
                colors[node] = "lightgray"
            G.add_node(node, label=label)

        # Add edges
        for parent, child in self.get_edges():
            G.add_edge(parent, child)

        for n in G.nodes:
            G.nodes[n]['rankdir'] = 'TB' if direction == "TB" else 'LR'
        G.graph['graph'] = {'ranksep': str(layer_sep), 'nodesep': str(node_sep)}

        # Hierarchical layout
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        if direction == "LR":
            pos = {k: (y, -x) for k, (x, y) in pos.items()}
        
        labels = nx.get_node_attributes(G, 'label')
        node_colors = [colors[n] for n in G.nodes]

        plt.figure(figsize=(16, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=4000,
            node_color=node_colors,
            font_size=7,
            font_color="black",
            edgecolors="k",
            linewidths=0.7,
            arrows=True,
            arrowsize=12
        )
        plt.title("Probabilistic Circuit Tree Visualization", fontsize=16)
        if save_path is not None:
            save_path = os.path.join(save_path, "pc_tree.png")
            plt.savefig(save_path, bbox_inches='tight')
            print_info(f"PC visualization saved to {save_path}")
# ============================================================
# PCNet
# ============================================================
class PCNet(ProbabilisticCircuit):
    def __init__(
        self, input_size=(256, 256), n_classes=2, distribution=None, device="cpu",
        max_depth=4, max_branching=3,
        seed=42, cv_module = None
    ):
        super().__init__(n_classes, device)
        self.input_size = input_size
        self.distribution = distribution
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        self.cv_module = cv_module
        random.seed(self.seed)

    # --------------------------------------------------------
    # Network structure helpers
    # --------------------------------------------------------
    def pairwise(self, nodes):
        """Group nodes pairwise for ProductNode"""
        random.shuffle(nodes)
        pairs = [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2 == 1:
            pairs.append((nodes[-1],))
        return pairs

    def prod_layer(self, current_nodes):
        """Build a ProductNode layer"""
        next_nodes = []
        for pair in self.pairwise(current_nodes):
            if len(pair) == 1:
                next_nodes.append(pair[0])
                continue
            prod = ProductNode(list(pair))
            self.add_node(prod)
            for child in pair:
                self.add_edge(prod, child)
            next_nodes.append(prod)
        return next_nodes

    def branchwise(self, nodes):
        n = len(nodes)
        if n <= 1:
            return [(nodes[0],)]  # just one singleton branch

        groups = []
        max_b = self.max_branching
        n_groups = max(1, n // random.randint(2, max_b) + random.randint(0, 2))

        for _ in range(n_groups):
            upper = min(max_b, n)
            if upper < 2:  # avoid invalid randint range
                groups.append(tuple(nodes))
                continue
            gsize = random.randint(2, upper)
            group = tuple(random.sample(nodes, gsize))
            groups.append(group)

        # ensure all nodes appear at least once
        covered = {x for g in groups for x in g}
        missing = [x for x in nodes if x not in covered]
        for m in missing:
            groups[random.randrange(len(groups))] += (m,)

        return groups


    def sum_layer(self, current_nodes):
        """Build a SumNode layer"""
        next_nodes = []
        for branch in self.branchwise(current_nodes):
            weights = torch.rand(len(branch), device=self.device)
            weights = weights / weights.sum()
            node = SumNode(list(branch), weights=weights, device=self.device)
            self.add_node(node)
            for child in branch:
                self.add_node(child)
                self.add_edge(node, child)
            next_nodes.append(node)
        return next_nodes
    
    def reduce_to_n(self, nodes, n_target):
        """
        Force reduce a list of nodes to exactly n_target nodes by grouping.
        """
        if len(nodes) == n_target:
            return nodes
        if len(nodes) < n_target:
            # Pad by duplicating last node
            while len(nodes) < n_target:
                nodes.append(nodes[-1])
            return nodes

        # Too many: group nodes into n_target groups
        groups = np.array_split(nodes, n_target)
        reduced = []
        for g in groups:
            if len(g) == 1:
                reduced.append(g[0])
            else:
                weights = torch.rand(len(g), device=self.device)
                weights = weights / weights.sum()
                node = SumNode(list(g), weights=weights, device=self.device)
                self.add_node(node)
                for child in g:
                    self.add_edge(node, child)
                reduced.append(node)
        return reduced


    def build_network(self, current_nodes):
        nodes = current_nodes
        for depth in range(1, self.max_depth + 1):
            if depth == self.max_depth:
                # Force reduce/pad to exactly n_classes
                nodes = self.reduce_to_n(nodes, self.n_classes)

                print(f"[DEBUG] Final children = {len(nodes)}, expected = {self.n_classes}")
                root = ClassifierNode(nodes)
                self.add_node(root)
                for child in nodes:
                    self.add_edge(root, child)
                return root

            nodes = self.prod_layer(nodes) if depth % 2 == 0 else self.sum_layer(nodes)
        


    def init_network(self, inputs, labels):
        """Initialize the PCNet structure"""
        input_nodes = self.set_input_nodes(inputs, labels)
        root = self.build_network(input_nodes)
        self.set_root(root)
        self.params = []
        for n in self.get_nodes():
            if isinstance(n, SumNode) or isinstance(n, InputNode):
                self.params += n.get_learnable_params()
        print_configs(f"Initialized PCNet with {len(self.get_nodes())} nodes, "
                      f"{sum(p.numel() for p in self.params)} trainable parameters")

    # --------------------------------------------------------
    # Training & Prediction
    # --------------------------------------------------------
    def set_input_nodes(self, inputs, labels):
        print_configs("Inputs tensor shape:", inputs.shape)
        B, C, H, W = inputs.shape

        nodes = []
        for i in range(C):
            node = InputNode(i, device=self.device)
            node.fit(inputs)
            nodes.append(node)
            self.add_node(node)
        return nodes
    
    def evaluate(self, x):
        # x should be (C, H, W), convert to (1, C, H, W)
        if self.cv_module is not None:
            x = [self.cv_module.get_output(x_, return_map=True) for x_ in x]
            x = np.stack(x, axis=0)
            x = torch.from_numpy(x).float().to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = self.root.evaluate(x)
        return out


    def predict(self, inputs):

        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()
        
    def state_dict(self):
        """
        Return a unified state_dict collecting parameters from all node modules.
        This mimics nn.Module.state_dict().
        """
        state = {}
        for node in self.get_nodes():
            if isinstance(node, nn.Module):
                for name, param in node.named_parameters():
                    key = f"{node.__class__.__name__}_{id(node)}.{name}"
                    state[key] = param.detach().clone()
        return state

    def load_state_dict(self, state_dict, strict=False):
        """
        Load parameters into the PCNet nodes.
        Works even if the structure has changed slightly (strict=False).
        """
        loaded, missing = 0, 0
        for node in self.get_nodes():
            if isinstance(node, nn.Module):
                for name, param in node.named_parameters():
                    key = f"{node.__class__.__name__}_{id(node)}.{name}"
                    if key in state_dict:
                        try:
                            with torch.no_grad():
                                src = state_dict[key].to(param.device)
                                if src.shape == param.shape:
                                    param.copy_(src)
                                else:
                                    # partial reuse if possible
                                    min_len = min(param.numel(), src.numel())
                                    param.view(-1)[:min_len].copy_(src.view(-1)[:min_len])
                                    print_debugging(f"⚠️ Resized param {key}: old {tuple(src.shape)} → new {tuple(param.shape)}")
                            loaded += 1
                        except Exception:
                            missing += 1
                    elif not strict:
                        # try fuzzy match by name only (ignoring id)
                        for k, v in state_dict.items():
                            if k.endswith(f".{name}"):
                                with torch.no_grad():
                                    param.copy_(v.to(param.device))
                                loaded += 1
                                break
                        else:
                            missing += 1
        print_debugging(f"Loaded {loaded} parameters, {missing} missing.")
        return {"loaded": loaded, "missing": missing}

    def named_parameters(self):
        """
        Generator yielding (name, param) for all sub-node parameters.
        """
        for node in self.get_nodes():
            if isinstance(node, nn.Module):
                for name, param in node.named_parameters():
                    yield f"{node.__class__.__name__}_{id(node)}.{name}", param
        
    def save_dict(self):
        """
        Return a compact summary dictionary of the PCNet model.
        This includes metadata, key stats, and memory usage in KB.
        """
        info = {}

        # --- Basic info ---
        info["model_type"] = self.__class__.__name__
        info["device"] = str(self.device)
        info["n_classes"] = self.n_classes
        info["input_size"] = tuple(self.input_size)
        info["max_depth"] = self.max_depth
        info["max_branching"] = self.max_branching
        info["seed"] = self.seed

        # --- Structure statistics ---
        nodes = list(self.get_nodes())
        edges = list(self.get_edges())
        info["n_nodes"] = len(nodes)
        info["n_edges"] = len(edges)

        type_counts = Counter([n.__class__.__name__ for n in nodes])
        info["node_types"] = dict(type_counts)

        # --- Parameter summary ---
        all_params = []
        mus, sigmas, nus, gates = [], [], [], []
        sum_weights = []
        input_feats = []

        for node in nodes:
            if isinstance(node, InputNode):
                input_feats.append(node.feature_name)
                mus.append(node.mu.item())
                sigmas.append(torch.exp(node.log_sigma).item())
                nus.append(torch.exp(node.log_nu).item())
                gates.append(torch.sigmoid(node.gate).item())
            elif isinstance(node, SumNode):
                w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()
                sum_weights.extend(w.tolist())

        def safe_stats(x):
            if len(x) == 0:
                return {"mean": None, "std": None, "min": None, "max": None}
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        info["parameter_summary"] = {
            "mu": safe_stats(mus),
            "sigma": safe_stats(sigmas),
            "nu": safe_stats(nus),
            "sum_weights": safe_stats(sum_weights),
        }

        info["n_parameters"] = sum(p.numel() for p in getattr(self, "params", []))
        info["input_features"] = input_feats

        # --- Memory usage (KB) ---
        model_memory_kb = sum(p.numel() * p.element_size() for p in self.params) / 1024
        info["memory_kb"] = model_memory_kb

        return info
    
    def __repr__(self):
        """
        Return a detailed string representation of the PCNet,
        including architecture info and hierarchical tree structure.
        """
        if not hasattr(self, "graph") or not self.root:
            return "<Uninitialized PCNet>"

        nodes = list(self.get_nodes())

        # Count node types
        counts = {
            "InputNode": sum(isinstance(n, InputNode) for n in nodes),
            "SumNode": sum(isinstance(n, SumNode) for n in nodes),
            "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
            "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
        }

        # Count parameters
        total_params = 0
        param_summary = {}
        for n in nodes:
            if isinstance(n, InputNode):
                p = sum(p.numel() for p in [n.mu, n.log_sigma, n.log_nu, n.logits, n.gate])
                param_summary["InputNode"] = param_summary.get("InputNode", 0) + p
            elif isinstance(n, SumNode):
                p = n.weights.numel()
                param_summary["SumNode"] = param_summary.get("SumNode", 0) + p
            total_params += p if "p" in locals() else 0

        # ---------------------------------------------------------------------
        # HEADER SECTION
        # ---------------------------------------------------------------------
        header = [
            "─────────────────────────────────────────────",
            "🧠  Probabilistic Circuit Network Summary",
            "─────────────────────────────────────────────",
            f"📦 Classes: {self.n_classes}",
            f"📏 Input size: {self.input_size}",
            f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}",
            f"💻 Device: {self.device}",
            "─────────────────────────────────────────────",
            "📊 Node counts:",
        ]
        for k, v in counts.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append(f"🧩 Trainable parameters: {total_params}")
        for k, v in param_summary.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append("📚 Structure:")
        header_text = "\n".join(header)
         # --- Memory usage (KB) ---
        model_memory_kb = sum(p.numel() * p.element_size() for p in self.params) / 1024
        header_text += f"Memory usage: {model_memory_kb:.2f} KB\n"

        # ---------------------------------------------------------------------
        # TREE SECTION
        # ---------------------------------------------------------------------
        lines = []

        def format_node(node, prefix="", is_last=True):
            connector = "└─ " if is_last else "├─ "

            # Node label
            if isinstance(node, ClassifierNode):
                label = f"ClassifierNode(classes={len(node.children)})"
            elif isinstance(node, SumNode):
                label = f"SumNode(children={len(node.children)}, weights={len(node.weights)})"
            elif isinstance(node, ProductNode):
                label = f"ProductNode(children={len(node.children)})"
            elif isinstance(node, InputNode):
                label = f"InputNode({node.feature_name})"
            else:
                label = node.__class__.__name__

            lines.append(f"{prefix}{connector}{label}")

            # Recurse for children
            if hasattr(node, "children") and node.children:
                new_prefix = prefix + ("   " if is_last else "│  ")
                for i, child in enumerate(node.children):
                    format_node(child, new_prefix, i == len(node.children) - 1)

        format_node(self.root)

        tree_text = "\n".join(lines)
        return f"{header_text}\n{tree_text}\n─────────────────────────────────────────────"

    
    def info(self, show_layers=True, show_params=True, show_counts=True):
        """
        Print a structured summary of the PCNet architecture.

        Args:
            show_layers (bool): show hierarchical layer summary
            show_params (bool): show learnable parameters per node type
            show_counts (bool): show node counts per type
        """
        print("─────────────────────────────────────────────")
        print("🧠  Probabilistic Circuit Network Summary")
        print("─────────────────────────────────────────────")
        print(f"📦 Classes: {self.n_classes}")
        print(f"📏 Input size: {self.input_size}")
        print(f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}")
        print(f"💻 Device: {self.device}")
        print("─────────────────────────────────────────────")

        nodes = list(self.get_nodes())
        total_params = 0

        if show_counts:
            counts = {
                "InputNode": sum(isinstance(n, InputNode) for n in nodes),
                "SumNode": sum(isinstance(n, SumNode) for n in nodes),
                "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
                "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
            }
            print("📊 Node counts:")
            for k, v in counts.items():
                print(f"  • {k:<15}: {v}")
            print("─────────────────────────────────────────────")

        if show_params:
            param_summary = []
            for n in nodes:
                if isinstance(n, InputNode):
                    p = sum(p.numel() for p in [n.mu, n.log_sigma, n.log_nu, n.logits])
                    param_summary.append(("InputNode", p))
                elif isinstance(n, SumNode):
                    p = n.weights.numel()
                    param_summary.append(("SumNode", p))
            total_params = sum(p for _, p in param_summary)
            print(f"🧩 Trainable parameters: {total_params}")
            by_type = {}
            for t, p in param_summary:
                by_type[t] = by_type.get(t, 0) + p
            for t, p in by_type.items():
                print(f"  • {t:<15}: {p}")
            print("─────────────────────────────────────────────")
         # --- Memory usage (KB) ---
        model_memory_kb = sum(p.numel() * p.element_size() for p in self.params) / 1024
        print(f"📚 Memory usage: {model_memory_kb:.2f} KB")
        print("─────────────────────────────────────────────")

        return {
            "n_nodes": len(nodes),
            "n_params": total_params,
            "depth": self.max_depth,
            "branching": self.max_branching,
        }

# ============================================================
# MultiPCNet: multiple subnetworks (experts) + learned selector
# Plug-and-play for your existing PCNet / Node classes
# ------------------------------------------------------------
# What you need already defined elsewhere in your codebase:
#   - Node, InputNode, SumNode, ProductNode, ClassifierNode
#   - ProbabilisticCircuit, PCNet (with init_network/evaluate/predict, etc.)
#   - print_configs, print_debugging, print_info utilities (optional)
# This file adds:
#   - SelectorNode (soft and hard variants)
#   - SubPCNet (thin wrapper around PCNet)
#   - MultiPCNet (K experts + selector as the new root)
# ============================================================
from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# If your project puts these in a package/module, import from there instead
# from your_project.pcnet import PCNet, ProbabilisticCircuit, Node, ClassifierNode


# ================================
# Selector nodes
# ================================
class SelectorNode(nn.Module, Node):
    """
    Soft mixture selector over sub-circuits (experts).
    Each child must return (B, C, H, W) *log-probabilities* (ClassifierNode output).

    evaluate(x) returns the log-mixture: log sum_i w_i * p_i(x)

    Notes
    -----
    - Fully differentiable; recommended for training.
    - The mixing weights (logits) are input-agnostic (global). If you want
      input-conditional gating, see InputGatedSelectorNode below.
    """
    def __init__(self, children: List[Node], device: str = "cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=children)
        self.device = device
        self.logits = nn.Parameter(torch.zeros(len(children), device=device))

    def get_learnable_params(self):
        return [self.logits]

    @property
    def n_experts(self):
        return len(self.children)

    def evaluate(self, x):
        # children[i].evaluate(x): (B, C, H, W) in log-space
        expert_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)  # (K, B, C, H, W)
        log_w = torch.log_softmax(self.logits, dim=0).view(-1, 1, 1, 1, 1)        # (K,1,1,1,1)
        return torch.logsumexp(log_w + expert_logs, dim=0)  # (B,C,H,W)


class HardBestSelectorNode(Node):
    """
    Non-differentiable selector: picks the expert with the highest average
    log-likelihood for the current batch. Use for evaluation/inference.

    Warning: Not suitable for gradient-based training.
    """
    def __init__(self, children: List[Node]):
        super().__init__(children=children)

    def evaluate(self, x):
        # Stack expert outputs and pick the best expert per batch
        # Strategy: average log-prob across classes and spatial dims
        expert_logs = [c.evaluate(x) for c in self.children]  # list of (B,C,H,W)
        stacked = torch.stack(expert_logs, dim=0)             # (K,B,C,H,W)
        # Compute a score per (K,B)
        scores = stacked.mean(dim=(2, 3, 4))                  # (K,B)
        best_k = scores.argmax(dim=0)                         # (B,)
        # Gather the chosen expert output per batch item
        B = stacked.size(1)
        out = []
        for b in range(B):
            out.append(stacked[best_k[b], b])                 # (C,H,W)
        return torch.stack(out, dim=0)                        # (B,C,H,W)


class InputGatedSelectorNode(nn.Module, Node):
    """
    Optional: input-dependent gating (tiny MLP over global-pooled inputs).
    - Provide x_features(x): (B, F) descriptor of the input (you can plug CV module outputs).
    - Learns p(expert | x) and mixes experts in log-space.
    """
    def __init__(self, children: List[Node], feat_dim: int, hidden: int = 64, device: str = "cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=children)
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, len(children))
        ).to(device)

    def get_learnable_params(self):
        return list(self.mlp.parameters())

    def evaluate_with_features(self, x, x_feat: torch.Tensor):
        # x_feat: (B, F)
        expert_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)  # (K,B,C,H,W)
        B = expert_logs.size(1)
        logits = self.mlp(x_feat)                         # (B, K)
        log_w = torch.log_softmax(logits, dim=1).transpose(0,1).view(-1,B,1,1,1)  # (K,B,1,1,1)
        return torch.logsumexp(log_w + expert_logs, dim=0)


# ================================
# SubPCNet: thin wrapper around your PCNet
# ================================
class SubPCNet:
    """
    Minimal wrapper to reuse your existing PCNet as an expert.
    Each sub-net owns its own graph, parameters, etc.
    """
    def __init__(self, base_pcnet_cls, *,
                 input_size=(256,256), n_classes=2, device="cpu",
                 max_depth=3, max_branching=3, seed=42, cv_module=None,
                 name: Optional[str] = None):
        self.model = base_pcnet_cls(
            input_size=input_size, n_classes=n_classes, device=device,
            max_depth=max_depth, max_branching=max_branching, seed=seed,
            cv_module=cv_module
        )
        self.name = name or f"SubPCNet_{id(self)}"
        self.device = device
        self.n_classes = n_classes

    # Proxy the Node API expected by selector
    def evaluate(self, x):
        return self.model.evaluate(x)  # (B,C,H,W) in log-space

    # Lifecycle helpers
    def init_network(self, inputs, labels):
        self.model.init_network(inputs, labels)

    def parameters(self):
        # Return generator over model params (as stored in self.model.params or via named_parameters)
        # Prefer the explicit param list your PCNet builds
        if hasattr(self.model, "params") and len(getattr(self.model, "params", [])):
            for p in self.model.params:
                yield p
        else:
            for _, p in self.model.named_parameters():
                yield p

    def save_dict(self):
        info = self.model.save_dict()
        info["subnet_name"] = self.name
        return info


# ================================
# MultiPCNet: K experts + selector root
# ================================
class MultiPCNet:
    """
    Build K sub-PCNets (experts) and a selector on top.

    selector_mode:
        - "soft":    SelectorNode with global learned mixture weights (default)
        - "hard":    HardBestSelectorNode (non-differentiable; use for eval only)
        - "gated":   InputGatedSelectorNode (needs x_features function)

    Bootstrapping:
        - Optionally initialize each expert on a bootstrapped subset of the training batch
          to encourage specialization (set bootstrap=True).
    """
    def __init__(self,
                 base_pcnet_cls,
                 *,
                 n_experts: int = 3,
                 input_size: Tuple[int,int] = (256,256),
                 n_classes: int = 2,
                 device: str = "cpu",
                 max_depth: int = 3,
                 max_branching: int = 3,
                 cv_module=None,
                 selector_mode: Literal["soft","hard","gated"] = "soft",
                 x_feature_fn=None,   # callable: tensor(B,C,H,W)-> tensor(B,F)
                 x_feature_dim: Optional[int] = None,
                 seed: int = 42,
                 bootstrap: bool = False,
                 name: str = "MultiPCNet"):
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.selector_mode = selector_mode
        self.x_feature_fn = x_feature_fn
        self.bootstrap = bootstrap

        # Build experts
        self.experts: List[SubPCNet] = []
        for k in range(n_experts):
            self.experts.append(
                SubPCNet(
                    base_pcnet_cls,
                    input_size=input_size,
                    n_classes=n_classes,
                    device=device,
                    max_depth=max_depth,
                    max_branching=max_branching,
                    seed=seed + k,
                    cv_module=cv_module,
                    name=f"Expert_{k}"
                )
            )

        # Build selector
        if selector_mode == "soft":
            self.selector = SelectorNode([e for e in self.experts], device=device)
        elif selector_mode == "hard":
            self.selector = HardBestSelectorNode([e for e in self.experts])
        elif selector_mode == "gated":
            assert x_feature_fn is not None and x_feature_dim is not None, "gated selector requires x_feature_fn and x_feature_dim"
            self.selector = InputGatedSelectorNode([e for e in self.experts], feat_dim=x_feature_dim, device=device)
        else:
            raise ValueError(f"Unknown selector_mode: {selector_mode}")

        # Collect trainable parameters
        self.params: List[torch.nn.Parameter] = []

    # -----------------------------
    # Initialization
    # -----------------------------
    def init_network(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        Initialize each expert PCNet. You can optionally bootstrap the inputs
        so each expert sees a slightly different subset.

        inputs: (B, C, H, W)
        labels: whatever your PCNet.init_network expects (kept for API symmetry)
        """
        B = inputs.shape[0]

        for k, expert in enumerate(self.experts):
            if self.bootstrap and B >= 4:
                # Sample with replacement ~80% of batch
                idx = torch.randint(0, B, (max(1, int(0.8*B)),), device=inputs.device)
                sub_in = inputs[idx]
                sub_lb = labels[idx] if isinstance(labels, torch.Tensor) and labels.shape[0] == B else labels
            else:
                sub_in, sub_lb = inputs, labels
            expert.init_network(sub_in, sub_lb)

        # Pack params from experts and selector
        self.params = []
        for e in self.experts:
            for p in e.parameters():
                self.params.append(p)
        if isinstance(self.selector, nn.Module):
            self.params += list(self.selector.get_learnable_params()) if hasattr(self.selector, 'get_learnable_params') else list(self.selector.parameters())

        try:
            n_params = sum(p.numel() for p in self.params if p.requires_grad)
            print_configs(f"Initialized MultiPCNet with {len(self.experts)} experts; trainable params: {n_params}")
        except Exception:
            pass

    # -----------------------------
    # Forward / evaluate / predict
    # -----------------------------
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, n_classes, H, W) in log-space."""
        if self.selector_mode == "gated":
            # Compute features for gating
            with torch.no_grad():
                x_feat = self.x_feature_fn(x)                    # (B,F)
            return self.selector.evaluate_with_features(x, x_feat)
        else:
            return self.selector.evaluate(x)

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()

    # -----------------------------
    # Training helpers
    # -----------------------------
    def named_parameters(self):
        # Yield names that reflect expert indices and selector
        for k, e in enumerate(self.experts):
            # Prefer the explicit params list to keep consistency with PCNet
            if hasattr(e.model, "params") and len(getattr(e.model, "params", [])):
                for i, p in enumerate(e.model.params):
                    yield f"experts.{k}.param_{i}", p
            else:
                for name, p in e.model.named_parameters():
                    yield f"experts.{k}.{name}", p
        if isinstance(self.selector, nn.Module):
            # Named parameters for selector
            for name, p in self.selector.named_parameters():
                yield f"selector.{name}", p

    def state_dict(self):
        state = {}
        # Experts
        for k, e in enumerate(self.experts):
            # Snapshot all parameters from each node/module via PCNet.state_dict()
            sd = e.model.state_dict()
            for key, tensor in sd.items():
                state[f"experts.{k}.{key}"] = tensor
        # Selector
        if isinstance(self.selector, nn.Module):
            for name, p in self.selector.named_parameters():
                state[f"selector.{name}"] = p.detach().clone()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        loaded, missing = 0, 0
        # Experts
        for k, e in enumerate(self.experts):
            # Extract subset for this expert
            prefix = f"experts.{k}."
            sub = {k2[len(prefix):]: v for k2, v in state_dict.items() if k2.startswith(prefix)}
            stats = e.model.load_state_dict(sub, strict=strict)
            loaded += stats.get("loaded", 0)
            missing += stats.get("missing", 0)
        # Selector
        if isinstance(self.selector, nn.Module):
            for name, p in self.selector.named_parameters():
                key = f"selector.{name}"
                if key in state_dict:
                    with torch.no_grad():
                        src = state_dict[key].to(p.device)
                        if src.shape == p.shape:
                            p.copy_(src)
                            loaded += 1
                        else:
                            # partial copy fallback
                            n = min(src.numel(), p.numel())
                            p.view(-1)[:n].copy_(src.view(-1)[:n])
                            loaded += 1
                else:
                    missing += 1
        try:
            print_debugging(f"MultiPCNet load_state: loaded={loaded}, missing={missing}")
        except Exception:
            pass
        return {"loaded": loaded, "missing": missing}

    # -----------------------------
    # Reporting
    # -----------------------------
    def save_dict(self):
        info = {
            "model_type": "MultiPCNet",
            "name": self.name,
            "device": self.device,
            "n_classes": self.n_classes,
            "n_experts": len(self.experts),
            "selector_mode": self.selector_mode,
        }
        info["experts"] = [e.save_dict() for e in self.experts]
        if isinstance(self.selector, SelectorNode):
            with torch.no_grad():
                w = torch.softmax(self.selector.logits, dim=0).detach().cpu().numpy().tolist()
            info["selector_weights"] = w
        return info


# ================================
# Convenience trainer (optional)
# ================================
class MultiPCNetTrainer:
    """
    Minimal training loop hook. Plug in your own loss & dataloaders.
    Assumes outputs are *logits* over classes per pixel (B,C,H,W) and uses
    pixel-wise cross-entropy by default.
    """
    def __init__(self, model: MultiPCNet, lr: float = 1e-3, weight_decay: float = 0.0):
        self.model = model
        # Build optimizer over collected parameters
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits in log-space; convert to prob via log_softmax? they should already be log-probs.
        # Targets expected (B,1,H,W) or (B,H,W) with class indices.
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        return F.nll_loss(F.log_softmax(logits, dim=1), targets)

    def step(self, x: torch.Tensor, y: torch.Tensor):
        self.opt.zero_grad()
        logits = self.model.evaluate(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_batch(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.model.evaluate(x)
        if y.ndim == 4 and y.size(1) == 1:
            y_ = y.squeeze(1)
        else:
            y_ = y
        pred = logits.argmax(dim=1)
        acc = (pred == y_).float().mean().item()
        return acc


# ================================
# Example wiring (usage)
# ================================
"""
from your_pcnet_impl import PCNet

# Build
mnet = MultiPCNet(
    base_pcnet_cls=PCNet,
    n_experts=3,
    input_size=(256,256),
    n_classes=2,
    device="cpu",
    max_depth=4,
    max_branching=3,
    selector_mode="soft",  # or "hard" / "gated"
    bootstrap=True,
)

# Initialize (pass a representative batch for InputNode fitting)
mnet.init_network(train_inputs, train_labels)

# Train with your loop or the convenience trainer
trainer = MultiPCNetTrainer(mnet, lr=1e-3)
for x, y in train_loader:
    loss = trainer.step(x, y)

# Inference
with torch.no_grad():
    out = mnet.evaluate(x)      # (B,C,H,W) log-probs
    pred = mnet.predict(x)      # numpy labels

# Introspection
print(mnet.save_dict())
"""


