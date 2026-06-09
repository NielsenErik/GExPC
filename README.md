# GExPC: An Explainable Evolutionary Approach for Crack Segmentation with Probabilistic Circuits

[![GECCO '26](https://img.shields.io/badge/GECCO-2026-blue)](https://doi.org/10.1145/3795101.3805281)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3795101.3805281-green)](https://doi.org/10.1145/3795101.3805281)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**Erik Nielsen** and **Giovanni Iacca** — University of Trento, Italy

> Presented at the Genetic and Evolutionary Computation Conference (GECCO Companion '26), July 13–17, 2026, San José, Costa Rica.

---

## Abstract

Crack segmentation is a fundamental task in structural inspection and has traditionally relied on resource-intensive, black-box Deep Learning (DL) models. To address this, we introduce **GExPC**, an explainable approach combining **Probabilistic Neural Circuits (PCNets)** with **Grammatical Evolution (GE)** to automatically discover compact, task-adapted architectures. Evaluated across six benchmarks, GExPC uses **fewer than 100 learnable parameters** yet achieves competitive performance — particularly in recall and shape preservation — against state-of-the-art DL models requiring millions of parameters. Our results demonstrate that evolutionary search over PC structures provides a highly efficient, interpretable alternative to standard DL for vision-based structural inspection.

---

## Results

Quantitative comparison (clIoU) across six benchmark datasets. GExPC results are reported as **mean ± std (best)** over five independent runs.

| Dataset | GExPC (mean ± std) | GExPC (best) | U-Net | U-Net++ | DeepCrack | GExPC Params |
|---|---|---|---|---|---|---|
| AEL | 0.39 ± .05 | 0.46 | 0.70 | 0.55 | 0.69 | **33** |
| Crack500 | 0.30 ± .05 | 0.35 | 0.49 | 0.52 | 0.53 | **37** |
| DeepCrack | 0.63 ± .01 | 0.64 | 0.81 | 0.86 | 0.82 | **39** |
| GAPS384 | 0.12 ± .04 | 0.17 | 0.48 | 0.49 | 0.55 | **35** |
| cracktree200 | 0.27 ± .07 | 0.34 | 0.59 | 0.00 | 0.00 | **39** |
| CrackSeg9k | 0.18 ± .09 | 0.24 | 0.48 | 0.49 | 0.55 | **32** |

> Baseline models (U-Net, U-Net++, DeepCrack) require ~30–36 million parameters. GExPC uses ≤100.

---

## Requirements

Python 3.10+ is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `torch >= 2.8.0`
- `torchvision >= 0.23.0`
- `numpy`, `opencv-python`, `scikit-image`, `matplotlib`
- `mlflow` (for experiment tracking)
- `graphviz` (for topology visualization)

---

## Datasets

GExPC is evaluated on six publicly available crack segmentation benchmarks:

| Dataset | Source |
|---|---|
| AEL, Crack500, DeepCrack, GAPS384, cracktree200 | [Omnicrack30k benchmark](https://github.com/benz2024/omnicrack30k) |
| CrackSeg9k | [Kaggle](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) |

Download CrackSeg9k via kaggle hub:

```python
import kagglehub
path = kagglehub.dataset_download("lakshaymiddha/crack-segmentation-dataset")
```

---

## Training

Experiments are configured via JSON config files. To run GExPC on a dataset:

```bash
python src/main.py <path/to/config.json> <seed>
```

For example, to run with seed 1:

```bash
python src/main.py configs/AEL_pcnet_ge.json 1
```

### HPC (PBS cluster)

To run all five seeds across all six datasets using PBS job arrays:

```bash
bash hpc_scripts/run_all_100.sh
```

Each job is submitted as a PBS array job (`-J 1-5`), running five independent seeds per dataset. The template script (`hpc_scripts/template.sh`) configures 2 nodes × 10 CPUs and a 60-hour wall time.

### Configuration

Key hyperparameters (GE phase):

| Parameter | Value |
|---|---|
| Generations | 10 |
| Population size | 20 |
| Genome length | 96 |
| Crossover rate | 0.9 |
| Mutation rate | 0.05 |
| Elite individuals | 2 |
| Max depth | 3 |
| Max branching | 3 |

Key hyperparameters (training):

| Parameter | Value |
|---|---|
| Epochs (GE phase) | 50 |
| Epochs (full train) | 80 |
| Training set (GE phase) | 20% |
| Validation split | 20% |
| Image size | 256 × 256 |

---

## Explainability

A key feature of GExPC is intrinsic model explainability. The discovered PCNet topologies contain only a handful of learnable parameters, and their weight distributions can be directly inspected. The circuit structure also allows tracing how input pixels are progressively transformed through probabilistic representations up to the final binary segmentation output.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{nielsen2026gexpc,
  author    = {Nielsen, Erik and Iacca, Giovanni},
  title     = {{GExPC}: An Explainable Evolutionary Approach for Crack Segmentation with Probabilistic Circuits},
  booktitle = {Genetic and Evolutionary Computation Conference Companion (GECCO Companion '26)},
  year      = {2026},
  month     = {July},
  address   = {San Jos\'{e}, Costa Rica},
  publisher = {ACM},
  doi       = {10.1145/3795101.3805281},
  isbn      = {979-8-4007-2488-6/2026/07},
}
```

---

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
