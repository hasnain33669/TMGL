# TMGL: Topology-enhanced Multi-level Graph Learning for Molecular Property Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

## Overview

**TMGL** (Topology-enhanced Multi-level Graph Learning) is a novel deep learning framework for molecular property prediction that leverages multi-level graph representations including node-level, motif-level, and graph-level features. The model incorporates:

- **Topology-enhanced message passing** using GIN with topological features
- **Motif extraction** for capturing local structural patterns
- **Multi-level contrastive learning** for better representation learning
- **Adaptive fusion mechanism** for combining local, motif, and global representations
- **Support for binary classification, multi-label classification, and regression tasks**


## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Install from source

```bash
git clone https://github.com/yourusername/TMGL-Molecular-Property-Prediction.git
cd TMGL-Molecular-Property-Prediction
pip install -e .
