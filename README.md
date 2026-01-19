# MARA – Modular Angular-Radial Attention

This repository contains the implementation of **MARA**, a spherical attention module integrated into the MACE framework for equivariant message passing on atomic systems.

## Overview

MARA introduces a spherical attention mechanism operating on a discretized spherical grid and is designed to enhance message passing in equivariant neural networks.
The module is integrated into the **RealAgnosticInteractionBlock** and **RealAgnosticResidualInteractionBlock** of MACE and returns both attention-weighted messages and the corresponding attention weights.

An overview of the model architecture is shown in the figure below.

![Model architecture](docs/architectural_sketch.png)

## Requirements

The implementation has been developed and tested with the following setup:

- Python **3.10** with PyTorch **2.8.0** and CUDA **12.8**
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```
- **torch-scatter 2.8.0+cu128** - scatter operator 
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

- **torch-harmonics 0.8.0**  – spherical attention implementation  
  https://github.com/NVIDIA/torch-harmonics
- **MACE / ACE suite** – equivariant message passing framework  
  https://github.com/ACEsuit/mace
- **cuEquivariance** – CUDA-optimized equivariant operations  
  https://github.com/NVIDIA/cuEquivariance

We recommend using a CUDA-enabled GPU for both training and inference.

## Spherical Attention Module

The spherical attention operates on a discretized spherical grid.
By default, we use a **4 × 8 grid resolution**, unless otherwise specified.

The module returns:
- attention-weighted messages
- the corresponding attention weights (for analysis and visualization)

## Integration into MACE

The module is integrated into the following MACE blocks:

- `RealAgnosticInteractionBlock`
- `RealAgnosticResidualInteractionBlock`

The message passing procedure is as follows:
```
m_ji = self.conv_tp(
    node_feats[edge_index[0]],
    edge_attrs,
    tp_weights
)

m_ji, att = self.spherical_attention(
    m_ji,
    positions,
    edge_index,
    edge_feats
)

message = scatter_sum(
    src=m_ji,
    index=edge_index[1],
    dim=0,
    dim_size=node_feats.shape[0]
)
```

## Training and Hardware

- Training was primarily performed on **NVIDIA H100 GPUs**
- The same hyperparameters as the original [MACE paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf) were used 
- Inference benchmarks were conducted on an **RTX 4090**
