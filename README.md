# AGCL: Adaptive Multi-View Graph Contrastive Learning for Robust Network Intrusion Detection

**USENIX Security '27 Artifact**

## Overview

This artifact contains the complete implementation of AGCL, a robust graph-based network intrusion detection framework that learns invariant graph representations from multiple complementary graph views under adversarial conditions.

## Paper Abstract

Graph-based network intrusion detection systems have recently demonstrated strong capability in modeling complex communication relationships among network entities. However, these systems remain vulnerable to adaptive adversaries that manipulate graph structures and node attributes to evade detection, while evolving network environments introduce distribution shifts that further degrade detection performance. Existing graph learning approaches typically rely on static graph assumptions and fixed representation learning strategies, limiting their robustness against adversarial manipulation and dynamic traffic evolution. To address these challenges, we propose Adaptive multi-view Graph Contrastive representation Learning (AGCL), a robust graph-based intrusion detection framework that jointly learns invariant graph representations from multiple complementary graph views under adversarial conditions. AGCL integrates adaptive multi-view graph construction, dynamic positive pair selection, and adversarially consistent contrastive optimization to preserve discriminative representations despite structural perturbations and evolving traffic distributions.

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, CPU mode supported)
- Docker (optional)

## Quick Start

### Using Docker (Recommended)

```bash
# Build Docker image
docker build -t agcl-artifact .

# Run training
docker run --gpus all -v $(pwd)/results:/workspace/results agcl-artifact

# Run experiments
docker run --gpus all -v $(pwd)/results:/workspace/results agcl-artifact python scripts/run_experiments.py
