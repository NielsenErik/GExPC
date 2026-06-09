import torch
import torch.nn.functional as F
import numpy as np
import random, math, os, cv2, time, gc
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
from probabilistic_circuits.probabilistic_circuits import *

# ============================================================
# --- Grammar specifications for tree/DAG generation ---
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
    children: List["TreeSpec"]

@dataclass
class GateSpec:
    kind: str
    children: List["TreeSpec"]

@dataclass
class ResidualSpec:
    kind: str
    children: List["TreeSpec"]

# ======= replace your TreeSpec line with this =======
TreeSpec = Union[LeafSpec, SumSpec, ProdSpec, GateSpec, ResidualSpec]


# ============================================================
# --- Mapper (Codons → DAG grammar) ---
# ============================================================

class GEMapper:
    """
    Grammatical Evolution mapper adapted for DAG-compatible PCNet.
    Sum nodes can now reuse children across different groups (branchwise overlap).
    """
    def __init__(self, max_wraps=2):
        self.max_wraps = max_wraps

    def map_model(self, codons: List[int], n_features: int, max_depth: int, max_branching: int):
        """Map a genome into multiple DAG-compatible trees (e.g. one per class)."""
        specs, idx, wraps = [], 0, 0
        for _ in range(2):  # binary classifier → 2 roots
            tree, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth=0)
            specs.append(tree)
        return specs

    # --------------------- internals ---------------------

    def _next(self, codons, idx, wraps):
        if idx >= len(codons):
            wraps += 1
            if wraps > self.max_wraps:
                # Instead of returning None, fallback to pseudo-random codon
                return random.randint(0, 255), idx, wraps, True
            idx = 0
        return codons[idx], idx + 1, wraps, False


    def _fallback_feature(self, idx, depth, n_features):
        if n_features <= 0: return 0
        return (depth * 17 + idx) % n_features

    def _expand(self, codons, idx, wraps, n_features, max_depth, max_branching, depth):
        if depth >= max_depth:
            f = self._fallback_feature(idx, depth, n_features)
            return LeafSpec("input", f), idx, wraps

        c, idx, wraps, exhausted = self._next(codons, idx, wraps)
        if exhausted:
            f = self._fallback_feature(idx, depth, n_features)
            return LeafSpec("input", f), idx, wraps

        node_type = c % 5  # 0=leaf, 1=sum, 2=prod, 3=gate, 4=residual

        # Leaf
        if node_type == 0:
            f, idx, wraps, exhausted2 = self._next(codons, idx, wraps)
            if exhausted2:
                f = self._fallback_feature(idx, depth, n_features)
            else:
                f = f % max(1, n_features)
            return LeafSpec("input", f), idx, wraps

        # Sum (allow overlapping children)
        elif node_type == 1:
            kcod, idx, wraps, _ = self._next(codons, idx, wraps)
            k = 2 + (kcod % max(1, max_branching - 1))
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            # overlap (duplicate a few children)
            if len(children) > 2 and random.random() < 0.4:
                dup = random.choice(children)
                children.insert(random.randint(0, len(children)), dup)
            return SumSpec("sum", len(children), children), idx, wraps

        # Product (now with overlapping children too)
        elif node_type == 2:
            n_child = 2 + (c % max(1, max_branching - 1))
            children = []
            for _ in range(n_child):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            # allow fan-out by duplicating references
            if len(children) > 2 and random.random() < 0.4:
                dup = random.choice(children)
                children.insert(random.randint(0, len(children)), dup)
            return ProdSpec("prod", children), idx, wraps

        # Gate (2 children)
        elif node_type == 3:
            a, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            b, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            return GateSpec("gate", [a, b]), idx, wraps

        # Residual (2 children)
        elif node_type == 4:
            base, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            sub,  idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            return ResidualSpec("res", [base, sub]), idx, wraps
        else:
            n_child = 2 + (c % max(1, max_branching - 1))
            children = []
            for _ in range(n_child):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            return ProdSpec("prod", children), idx, wraps


# ============================================================
# --- PC Builder (TreeSpec → PCNet DAG) ---
# ============================================================

class PCBuilder:
    """Builds a DAG-compatible PCNet from grammar specs."""
    def __init__(self, pcnet):
        self.pc = pcnet
        self.input_cache = {}
        self._memo = {}  # NEW: spec-id -> built PC node

    def _collect_features(self, t):
        """Recursively collect feature indices from a TreeSpec."""
        out = []

        def walk(x):
            if isinstance(x, LeafSpec):
                out.append(x.feature_idx)
            elif hasattr(x, "children"):
                for c in x.children:
                    walk(c)

        walk(t)
        return out
    def build_classifier(self, trees, inputs):
        feats = sorted(set(f for t in trees for f in self._collect_features(t)))
        for f in feats:
            if f not in self.input_cache:
                node = InputNode(f, device=self.pc.device)
                node.fit(inputs)
                self.input_cache[f] = node
                self.pc.add_node(node)

        # Build once per unique subtree thanks to memoization
        subcircuits = [self._build_subtree(t) for t in trees]

        root = ClassifierNode(subcircuits)
        self.pc.add_node(root)
        for c in subcircuits:
            self.pc.add_edge(root, c)
        self.pc.set_root(root)

        # collect params (unchanged)
        self.pc.params = []
        for n in self.pc.get_nodes():
            if isinstance(n, nn.Module):
                if hasattr(n, "get_learnable_params"):
                    self.pc.params += list(n.get_learnable_params())
                else:
                    for _, p in n.named_parameters(recurse=False):
                        if p.requires_grad:
                            self.pc.params.append(p)
        return root

    def _build_subtree(self, t):
        # --- memoization: reuse node instances for shared specs ---
        key = id(t)
        if key in self._memo:
            return self._memo[key]

        if isinstance(t, LeafSpec):
            node = self.input_cache[t.feature_idx]
            # Leaves are already shared via input_cache; still return early
            self._memo[key] = node
            return node

        elif isinstance(t, SumSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = SumNode(kids, weights=torch.rand(len(kids), device=self.pc.device), device=self.pc.device)
            self.pc.add_node(node)
            for c in kids:
                self.pc.add_edge(node, c)
            self._memo[key] = node
            return node

        elif isinstance(t, ProdSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = ProductNode(kids)
            self.pc.add_node(node)
            for c in kids:
                self.pc.add_edge(node, c)
            self._memo[key] = node
            return node

        elif isinstance(t, GateSpec):
            left, right = [self._build_subtree(c) for c in t.children]
            node = GateNode(left, right, device=self.pc.device)
            self.pc.add_node(node)
            self.pc.add_edge(node, left)
            self.pc.add_edge(node, right)
            self._memo[key] = node
            return node

        elif isinstance(t, ResidualSpec):
            base, sub = [self._build_subtree(c) for c in t.children]
            node = ResidualNode(base, sub, device=self.pc.device)
            self.pc.add_node(node)
            self.pc.add_edge(node, base)
            self.pc.add_edge(node, sub)
            self._memo[key] = node
            return node

        else:
            raise ValueError(f"Unknown spec type: {type(t)}")


# ============================================================
# --- GE Optimizer ---
# ============================================================

class UnsupervisedGEOptimizer:
    """
    Unsupervised GE optimizer for PCNet / MultiPCNet.
    Uses only the input X and an unsupervised criterion such as
    UnsupervisedPCNetLoss(logits, x).
    """
    def __init__(
        self,
        pop_size=16,
        genome_len=64,
        max_depth=4,
        max_branching=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elite_k=2,
        n_classes=1,
        device="cpu",
        seed=42,
        n_jobs=None,
        report=None,
        log_path=None,
        model_type="multi_pcnet",
        n_experts=5,
        selector_mode="soft",
    ):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_k = elite_k
        self.n_classes = n_classes
        self.device = device
        self.mapper = GEMapper(max_wraps=2)
        self.rng = np.random.default_rng(seed)
        self.n_jobs = n_jobs or os.cpu_count()
        self.report = report
        self.log_path = log_path
        self.evolution_report = os.path.join(log_path, "evolution_report.txt") if log_path else None
        if self.evolution_report:
            with open(self.evolution_report, "w") as f:
                f.write("Generation,Avg_Fitness,Best_Fitness,Avg_Complexity,Best_Complexity,All_Time_Best\n")

        self.model_type = model_type
        self.n_experts = n_experts
        self.selector_mode = selector_mode


    # -----------------------------------------------------
    # GA primitives
    # -----------------------------------------------------
    def init_population(self):
        return [
            self.rng.integers(0, 256, size=self.genome_len, dtype=np.int32).tolist()
            for _ in range(self.pop_size)
        ]

    def mutate(self, genome):
        return [
            random.randint(0, 255) if random.random() < self.mutation_rate else g
            for g in genome
        ]

    def crossover(self, g1, g2):
        if random.random() > self.crossover_rate:
            return g1[:], g2[:]
        cut = random.randint(1, len(g1) - 2)
        return g1[:cut] + g2[cut:], g2[:cut] + g1[cut:]


    # -----------------------------------------------------
    # UNSUPERVISED dataloader: only X
    # -----------------------------------------------------
    def get_dataloader(self, inputs, batch_size=32):
        dataset = TensorDataset(inputs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # -----------------------------------------------------
    # Fitness evaluation for a single genome
    # -----------------------------------------------------
    def _evaluate_single(self, genome, inputs, criterion, n_classes, train_steps, lr, seed_offset=0, global_index=None, param_bank=None):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        sig = hash(tuple(genome))

        random.seed(seed_offset)
        np.random.seed(seed_offset)
        torch.manual_seed(seed_offset)

        # ------ Map genome → trees ------
        trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)

        # ------ Build PCNet or MultiPCNet ------
        if self.model_type == "multi_pcnet":
            base_pc_cls = PCNet
            pc = MultiPCNet(
                base_pcnet_cls=base_pc_cls,
                n_experts=self.n_experts,
                input_size=inputs.shape[1:],
                n_classes=n_classes,
                device=self.device,
                max_depth=self.max_depth,
                max_branching=self.max_branching,
                selector_mode=self.selector_mode,
                bootstrap=True,
            )
            pc.init_network(inputs, None)   # UNSUPERVISED
        else:
            pc = PCNet(
                input_size=inputs.shape[1:], n_classes=n_classes, device=self.device,
                max_depth=self.max_depth, max_branching=self.max_branching
            )
            builder = PCBuilder(pc)
            builder.build_classifier(trees, inputs)

        # Reuse previous weights if available
        if param_bank is not None and sig in param_bank:
            try:
                pc.load_state_dict(param_bank[sig], strict=False)
            except:
                pass

        # ---------- Train/val split ----------
        split = int(0.8 * len(inputs))
        x_tr = inputs[:split].to(self.device)
        x_va = inputs[split:].to(self.device)

        opt = torch.optim.Adam(pc.params, lr=lr, weight_decay=1e-4)

        train_loader = self.get_dataloader(x_tr)
        val_loader = self.get_dataloader(x_va)

        # -------------------------
        # Training loop (UNSUPERVISED)
        # -------------------------
        for _ in range(train_steps):
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                logits = pc.evaluate(xb)
                loss = criterion(logits, xb)  # <--- UNSUPERVISED LOSS
                opt.zero_grad()
                loss.backward()
                opt.step()

        # -------------------------
        # Validation: also UNSUPERVISED
        # -------------------------
        with torch.no_grad():
            val_loss = 0.0
            for (xb,) in val_loader:
                xb = xb.to(self.device)
                logits = pc.evaluate(xb)
                val_loss += criterion(logits, xb).item()

            val_loss /= max(1, len(val_loader))

        # Save model state
        if param_bank is not None:
            param_bank[sig] = deepcopy(pc.state_dict())

        del pc, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return val_loss, global_index


    # -----------------------------------------------------
    # Worker wrapper
    # -----------------------------------------------------
    def _worker_eval(self, idx, genome, inputs, queue, criterion, n_classes, train_steps, lr, param_bank):
        try:
            loss, gid = self._evaluate_single(genome, inputs, criterion, n_classes, train_steps, lr, idx, idx, param_bank)
            queue.put((loss, gid))
        except Exception as e:
            print(f"[Worker {idx}] Error: {e}")
            queue.put((float("inf"), idx))


    # -----------------------------------------------------
    # Parallel population evaluation
    # -----------------------------------------------------
    def evaluate_population(self, population, inputs, criterion, n_classes=1, train_steps=20, lr=1e-2, param_bank=None):
        n_cpus = min(self.n_jobs, len(population))
        queues = [mp.Queue() for _ in range(len(population))]
        procs = []

        for i, genome in enumerate(population):
            p = mp.Process(target=self._worker_eval,
                           args=(i, genome, inputs, queues[i], criterion, n_classes, train_steps, lr, param_bank))
            p.start()
            procs.append(p)
            if len(procs) >= n_cpus:
                for p in procs: p.join()
                procs.clear()

        for p in procs:
            p.join()

        results = []
        for q in queues:
            try:
                results.append(q.get(timeout=30))
            except:
                results.append((float("inf"), len(results)))

        losses = [None] * len(population)
        for loss, idx in results:
            losses[idx] = loss

        return losses


    # -----------------------------------------------------
    # EVOLUTIONARY LOOP (UNSUPERVISED)
    # -----------------------------------------------------
    def evolve(self, inputs, criterion, generations=10, n_classes=1, train_steps=20, lr=1e-2):
        manager = Manager()
        param_bank = manager.dict()

        population = self.init_population()

        best_score = float("inf")
        best_genome = None
        history = []
        complexity_hist = []

        for gen in range(generations):
            print(f"🧬 Generation {gen+1}/{generations}")

            losses = self.evaluate_population(
                population, inputs, criterion,
                n_classes, train_steps, lr, param_bank
            )

            # Complexity measure
            complexities = []
            for genome in population:
                trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)

                dummy_pc = PCNet(
                    input_size=inputs.shape[1:], n_classes=n_classes,
                    device=self.device, max_depth=self.max_depth,
                    max_branching=self.max_branching
                )
                PCBuilder(dummy_pc).build_classifier(trees, inputs)
                complexities.append(sum(p.numel() for p in getattr(dummy_pc, "params", [])))

            # Pareto sort (loss first, complexity second)
            scored = list(zip(losses, complexities, population))
            scored.sort(key=lambda x: (x[0], x[1]))

            losses = [s[0] for s in scored]

            avg_loss = sum(losses) / len(losses)
            avg_complexity = sum(c for _, c, _ in scored) / len(scored)

            best_loss, best_complexity, best = scored[0]

            history.append(best_loss)
            complexity_hist.append(best_complexity)

            if best_loss <= best_score:
                best_score = best_loss
                best_genome = deepcopy(best)

            # Elitism
            elites = [deepcopy(g) for _, _, g in scored[:self.elite_k]]
            new_pop = elites[:]

            def select():
                i, j = random.randrange(len(scored)), random.randrange(len(scored))
                return scored[i] if (scored[i][0] < scored[j][0] or
                                     (scored[i][0] == scored[j][0] and scored[i][1] < scored[j][1])) else scored[j]

            # Build next population
            while len(new_pop) < self.pop_size:
                p1, p2 = select()[2], select()[2]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop

            print(f"✅ Gen {gen+1}: best loss={best_loss:.4f}, complexity={best_complexity}")

            if self.evolution_report:
                with open(self.evolution_report, "a") as f:
                    f.write(f"{gen+1},{avg_loss:.4f},{best_loss:.4f},{avg_complexity:.4f},{best_complexity},{best_score:.4f}\n")

        # Build final best PC
        trees = self.mapper.map_model(best_genome, inputs.shape[1], self.max_depth, self.max_branching)

        final_pc = PCNet(
            input_size=inputs.shape[1:], n_classes=n_classes,
            device=self.device, max_depth=self.max_depth, max_branching=self.max_branching
        )
        PCBuilder(final_pc).build_classifier(trees, inputs)

        return final_pc, best_genome, {"loss": history, "complexity": complexity_hist}
