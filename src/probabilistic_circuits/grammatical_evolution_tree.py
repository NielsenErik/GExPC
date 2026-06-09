# ============================================================
# Parallel HPC-optimized GE components
# ============================================================

from dataclasses import dataclass
from typing import List, Union
import random, math, numpy as np, torch, tqdm
from joblib import Parallel, delayed
from copy import deepcopy
import os
from probabilistic_circuits.probabilistic_circuits import *
from experiment import *
from utils import *
import torch.multiprocessing as mp
from multiprocessing import Manager
import time


def genome_signature(genome):
    return hash(tuple(genome))

# ============================================================
# Tree grammar specs
# ============================================================

@dataclass
class LeafSpec:
    kind: str
    feature_idx: int

@dataclass
class SumSpec:
    kind: str
    k: int
    children: List["TreeSpec"]

@dataclass
class ProdSpec:
    kind: str
    left: "TreeSpec"
    right: "TreeSpec"

TreeSpec = Union[LeafSpec, SumSpec, ProdSpec]


# ============================================================
# Mapper
# ============================================================

class GEMappingError(Exception): pass

class GEMapper:
    """Grammar-based mapper from codon sequence to TreeSpec (total/robust)."""
    def __init__(self, max_wraps=2):
        self.max_wraps = max_wraps

    def map_model(self, codons: List[int], n_features: int, max_depth: int, max_branching: int):
        specs, idx, wraps = [], 0, 0
        for _ in range(2):  # binary classifier = 2 trees
            tree, idx, wraps = self._expand_tree(
                codons, idx, wraps, n_features, max_depth, max_branching, depth=0
            )
            specs.append(tree)
        return specs

    # ---------- helpers ----------
    def _next_codon(self, codons, idx, wraps):
        """
        Returns (codon_or_None, next_idx, wraps, exhausted_flag).
        When codons are exhausted beyond max_wraps, codon is None and exhausted=True.
        """
        if idx >= len(codons):
            wraps += 1
            if wraps > self.max_wraps:
                # Do NOT raise; signal exhaustion to callers
                return None, idx, wraps, True
            idx = 0
        return codons[idx], idx + 1, wraps, False

    def _fallback_feature(self, idx, depth, n_features):
        # Deterministic fallback (no RNG) to keep runs reproducible
        if n_features <= 0:
            return 0
        return (depth * 31 + idx) % n_features

    # ---------- expansion ----------
    def _expand_tree(self, codons, idx, wraps, n_features, max_depth, max_branching, depth):
        # Depth cap ⇒ must be a leaf
        if depth >= max_depth:
            f, idx, wraps = self._pick_feature(codons, idx, wraps, n_features, depth)
            return LeafSpec("input", f), idx, wraps

        c, idx, wraps, exhausted = self._next_codon(codons, idx, wraps)
        if exhausted:
            # No more codons we can legally read → emit a leaf
            f = self._fallback_feature(idx, depth, n_features)
            return LeafSpec("input", f), idx, wraps

        node_type = c % 3  # 0: leaf, 1: sum, 2: prod

        if node_type == 0:  # leaf
            f, idx, wraps = self._pick_feature(codons, idx, wraps, n_features, depth)
            return LeafSpec("input", f), idx, wraps

        elif node_type == 1:  # sum
            kcod, idx, wraps, exhausted_k = self._next_codon(codons, idx, wraps)
            if exhausted_k:
                k = 2  # fallback
            else:
                k = 2 + (kcod % max(1, max_branching - 1))
            # Build children with increased depth; children count is bounded by k
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand_tree(
                    codons, idx, wraps, n_features, max_depth, max_branching, depth + 1
                )
                children.append(ch)
            return SumSpec("sum", k, children), idx, wraps

        else:  # prod
            left,  idx, wraps = self._expand_tree(
                codons, idx, wraps, n_features, max_depth, max_branching, depth + 1
            )
            right, idx, wraps = self._expand_tree(
                codons, idx, wraps, n_features, max_depth, max_branching, depth + 1
            )
            return ProdSpec("prod", left, right), idx, wraps

    def _pick_feature(self, codons, idx, wraps, n_features, depth):
        c, idx, wraps, exhausted = self._next_codon(codons, idx, wraps)
        if exhausted:
            return self._fallback_feature(idx, depth, n_features), idx, wraps
        return c % n_features, idx, wraps



# ============================================================
# PC Builder
# ============================================================

class PCBuilder:
    """Parallel-safe circuit builder from TreeSpec."""
    def __init__(self, pcnet):
        self.pc = pcnet
        self.input_cache = {}

    def build_classifier(self, trees, inputs):
        feats = sorted(set(f for t in trees for f in self._collect_features(t)))
        # Shared initialization of input nodes
        for f in feats:
            if f not in self.input_cache:
                node = InputNode(f, device=self.pc.device)
                node.fit(inputs)
                self.input_cache[f] = node
                self.pc.add_node(node)

        subcircuits = [self._build_subtree(t) for t in trees]
        root = ClassifierNode(subcircuits)
        self.pc.add_node(root)
        for c in subcircuits:
            self.pc.add_edge(root, c)
        self.pc.set_root(root)

        # collect params
        params = []
        for n in self.pc.get_nodes():
            if isinstance(n, SumNode):
                params.append(n.weights)
            elif isinstance(n, InputNode):
                params.extend([n.mu, n.log_sigma, n.log_nu, n.logits, n.gate])
        self.pc.params = params
        return root

    def _collect_features(self, t):
        out = []
        def walk(x):
            if isinstance(x, LeafSpec): out.append(x.feature_idx)
            elif isinstance(x, SumSpec): [walk(c) for c in x.children]
            elif isinstance(x, ProdSpec): walk(x.left); walk(x.right)
        walk(t)
        return out

    def _build_subtree(self, t):
        if isinstance(t, LeafSpec):
            return self.input_cache[t.feature_idx]
        elif isinstance(t, SumSpec):
            kids = [self._build_subtree(c) for c in t.children]
            w = torch.rand(len(kids), device=self.pc.device)
            node = SumNode(kids,device=self.pc.device)
            self.pc.add_node(node)
            [self.pc.add_edge(node, c) for c in kids]
            return node
        elif isinstance(t, ProdSpec):
            left, right = self._build_subtree(t.left), self._build_subtree(t.right)
            node = ProductNode([left, right])
            self.pc.add_node(node)
            self.pc.add_edge(node, left)
            self.pc.add_edge(node, right)
            return node
        raise ValueError("Unknown TreeSpec.")


# ============================================================
# GE Optimizer (parallelized)
# ============================================================

class GEOptimizer:
    def __init__(
        self,
        pop_size=16,
        genome_len=64,
        max_depth=4,
        max_branching=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elite_k=2,
        device="cpu",
        seed=42,
        n_jobs=None,
        report=None
    ):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_k = elite_k
        self.device = device
        self.mapper = GEMapper(max_wraps=2)
        self.rng = np.random.default_rng(seed)
        self.n_jobs = n_jobs or os.cpu_count()
        self.report = report

    def _local_n_jobs(self):
        """Limit joblib workers per-rank to avoid oversubscription when using torchrun."""
        if dist.is_available() and dist.is_initialized():
            return 1  # Avoid loky in distributed ranks

        return self.n_jobs
    
    def cl_iou_score(self,pred_mask, true_mask, tau=2):
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

    # -----------------------------
    # Genetic operators
    # -----------------------------
    def init_population(self):
        return [self.rng.integers(0, 256, size=self.genome_len, dtype=np.int32).tolist()
                for _ in range(self.pop_size)]

    def mutate(self, genome):
        g = genome[:]
        for i in range(len(g)):
            if random.random() < self.mutation_rate:
                g[i] = random.randint(0, 255)
        return g

    def crossover(self, g1, g2):
        if random.random() > self.crossover_rate:
            return g1[:], g2[:]
        cut = random.randint(1, len(g1) - 2)
        return g1[:cut] + g2[cut:], g2[:cut] + g1[cut:]

    def get_dataloader(self, inputs, labels, batch_size):
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # -----------------------------
    # Fitness evaluation (loss only)
    # -----------------------------
    def _evaluate_single(self, genome, inputs, labels, criterion, n_classes, train_steps, lr, seed_offset=0, global_index=None, param_bank=None):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sig = hash(tuple(genome))
        torch.manual_seed(seed_offset)
        np.random.seed(seed_offset)
        random.seed(seed_offset)
        print_debugging(f"Genome {global_index} (pid {os.getpid()}) evaluation started.")
        with open(self.report, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genome {global_index} pid {os.getpid()} evaluation started.\n")
        trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)

        pc = PCNet(input_size=inputs.shape[1:], n_classes=n_classes, device=self.device,
                   max_depth=self.max_depth, max_branching=self.max_branching)
        builder = PCBuilder(pc)
        builder.build_classifier(trees, inputs)

        # Load existing weights if available
        if param_bank is not None and sig in param_bank:
            try:
                pc.load_state_dict(param_bank[sig], strict=False)
                print_debugging(f"Reused weights for genome {global_index}")
            except Exception as e:
                print_debugging(f"Weight load failed: {e}")

        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        optimizer = torch.optim.Adam(pc.params, lr=1e-3, weight_decay=1e-4)

        # Shuffle then split
        split = int(len(inputs) * 0.8)
        x_tr, y_tr = inputs[:split].to(self.device), labels[:split].to(self.device)
        x_va, y_va = inputs[split:].to(self.device), labels[split:].to(self.device)

        train_loader = self.get_dataloader(x_tr, y_tr, batch_size=32)
        val_loader = self.get_dataloader(x_va, y_va, batch_size=32)

        for _ in range(train_steps):
            for xb, yb in train_loader:
                logits = pc.evaluate(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            val_loss = 0.0
            cl_iou = 0.0
            for xb, yb in val_loader:
                logits = pc.evaluate(xb)
                val_loss += criterion(logits, yb).item()
                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
                true_mask = yb.unsqueeze(1).float()
                cl_iou += self.cl_iou_score(pred_mask, true_mask)
            val_loss /= max(1, len(val_loader))
            cl_iou = cl_iou/len(val_loader)

        if param_bank is not None:
            param_bank[sig] = deepcopy(pc.state_dict())

        del pc, builder, optimizer, train_loader, val_loader, x_tr, y_tr, x_va, y_va, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()
        
        print_info(f"    - Genome {global_index} loss: {val_loss:.4f}, clIoU: {cl_iou:.4f}, ")
        return val_loss, global_index
    
    def _worker_eval(self, idx, genome, input_data, label_data, queue, criterion, n_classes, train_steps, lr, param_bank):
            """Evaluate a single genome and push (loss, index) to queue."""
            try:
                loss, global_idx = self._evaluate_single(
                    genome=genome,
                    inputs=input_data,
                    labels=label_data,
                    criterion=criterion,
                    n_classes=n_classes,
                    train_steps=train_steps,
                    lr=lr,
                    seed_offset=idx,
                    global_index=idx,
                    param_bank=param_bank,
                )
                queue.put((loss, global_idx))
            except Exception as e:
                queue.put((float("inf"), idx))  # fail-safe: assign high loss
                print(f"[Worker {idx}] Error: {e}")

    def evaluate_population(
        self,
        population,
        inputs,
        labels,
        criterion,
        n_classes=2,
        train_steps=30,
        lr=1e-2,
        param_bank=None,
    ):
        """
        Multiprocessing-based population evaluation.
        Each genome is evaluated in an independent process.
        Returns a list of losses aligned with `population` order.
        """


        # ----------------------------------------------------------------------
        # Setup multiprocessing
        # ----------------------------------------------------------------------
        n_cpus = self.n_jobs
        n_cpus = min(n_cpus, len(population))
        queues = [mp.Queue() for _ in range(len(population))]
        print_info(f"🧬 Evaluating population of size {len(population)} using {n_cpus} parallel processes.")
        # ----------------------------------------------------------------------
        # Spawn processes
        # ----------------------------------------------------------------------
        processes = []
        for i, genome in enumerate(population):
            p = mp.Process(target=self._worker_eval, args=(i, genome, inputs, labels, queues[i], criterion, n_classes, train_steps, lr, param_bank))
            p.start()
            processes.append(p)

            # Optional: throttle number of concurrent processes
            if len(processes) >= n_cpus:
                for p in processes:
                    p.join()
                   
                processes.clear()

        # Wait for any remaining
        for p in processes:
            p.join()
        # ----------------------------------------------------------------------
        # Collect results
        # ----------------------------------------------------------------------
        results = []
        for q in queues:
            try:
                results.append(q.get(timeout=60))
            except Exception:
                results.append((float("inf"), len(results)))  # fallback if missing

        # Rebuild aligned losses list
        losses = [None] * len(population)
        for loss, idx in results:
            losses[idx] = loss

        print_info(f"🧬 Population evaluation complete — losses: {losses}")
        return losses


    # -----------------------------
    # Evolutionary loop
    # -----------------------------
    def evolve(self, train_inputs, train_labels, criterion, generations=10, n_classes=2, train_steps=20, lr=1e-2):
        """
        Runs evolution returning:
          - best_pc: the final rebuilt-and-trained PC (once, at the end)
          - best_genome: list[int]
          - history: list[float] of best loss per generation
        """
        manager = Manager()
        param_bank = manager.dict()
        population = self.init_population()
        history, best_score, best_genome = [], math.inf, None
        for gen in range(generations):
            print_info(f"🧬 Generation {gen+1}/{generations} (evaluating {len(population)} genomes in parallel)")
            losses = self.evaluate_population(population, train_inputs, train_labels, criterion,
                                              n_classes=n_classes, train_steps=train_steps, lr=lr, param_bank=param_bank)

            # pair (loss, genome), find best
            scored = list(zip(losses, population))
            scored.sort(key=lambda x: x[0])
            history.append(scored[0][0])

            if scored[0][0] < best_score:
                best_score = scored[0][0]
                best_genome = deepcopy(scored[-0][1])

            # elites
            elites = [deepcopy(s[1]) for s in scored[:min(self.elite_k, len(scored))]]
            new_pop = elites[:]

            # tournament selection helper
            def pick():
                i, j = random.randrange(len(scored)), random.randrange(len(scored))
                return scored[i] if scored[i][0] < scored[j][0] else scored[j]

            # fill population
            while len(new_pop) < self.pop_size:
                p1 = pick()[1]
                p2 = pick()[1]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop
            print_info(f"✅ Gen {gen+1}: best loss so far = {best_score:.4f}")
            with open(self.report, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Gen {gen+1}: best loss so far = {best_score:.4f}\n")

        # After evolution, rebuild + train best genome once and return that PC
        trees = self.mapper.map_model(best_genome, train_inputs.shape[1], self.max_depth, self.max_branching)

        best_pc = PCNet(input_size=train_inputs.shape[1:], n_classes=n_classes, device=self.device,
                   max_depth=self.max_depth, max_branching=self.max_branching)
        builder = PCBuilder(best_pc)
        builder.build_classifier(trees, train_inputs)
        # Optionally, you could append final_loss to history (it should be close to the best_score)
        return best_pc, best_genome, history
