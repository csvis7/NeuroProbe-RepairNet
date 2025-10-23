# NeuroProbe-RepairNet

NeuroProbe-RepairNet: An open-source toolkit for diagnosing and repairing deep learning training failures (exploding gradients, batch norm drift, dead channels) with automatic interventions.

# NeuroProbe-RepairNet

An open-source toolkit for diagnosing and repairing deep learning training failures.  
Focus: automatically detect issues like exploding gradients, batch norm drift, and dead channels — and apply safe, reversible fixes.

---

## 🚀 Roadmap

- **Phase 0 (Week 0):** Baseline setup (ResNet18 → CIFAR-10, logging, Colab notebook)
- **Phase 1 (Week 1):** Instrumentation (log activations, gradients, BN stats)
- **Phase 2 (Week 2–3):** Failure detectors (rule-based diagnostics)
- **Phase 3 (Week 4–6):** Repair primitives + auto-repair loop
- **Phase 4 (Week 7+):** Dashboard, advanced detectors, release

---

## 🧪 Phase 0: Baseline (Completed)
- Implemented reproducible ResNet-18 → CIFAR-10 pipeline in PyTorch
- Includes config system, experiment logging (Weights & Biases), checkpointing
- Achieved 57 % validation accuracy (5 epochs) — reproducible baseline
- Run it on Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csvis7/NeuroProbe-RepairNet/blob/main/notebooks/Resnet_baseline
- .ipynb)

