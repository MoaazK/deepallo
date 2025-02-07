# DeepAllo: Allosteric Site Prediction using Protein Language Model (pLM) with Multitask Learning
<!-- ![Results](https://github.com/MoaazK/comp511-project/blob/master/assets/result.png?raw=true) -->

Implementation and Inference pipeline for [DeepAllo](https://www.biorxiv.org/content/10.1101/2024.10.09.617427v1)

- [Quick Start](#quick-start)
- [Description](#description)
- [Training & Hardware](#training-hardware)
- [Results](#results)
- [Requirements](#requirements)
- [Data](#data)

## Results
| Model | F1 | Precision | Recall |
| ------------- | ------------- | ------------- | ------------- |
| DeepAllo<sub>*XGBoost*<sub> | 0.8870 | 0.9107 | 0.8644 |
| DeepAllo<sub>*AutoML*<sub> | **0.8966** | **0.9231** | **0.8814** |

## Requirements
1. [Anaconda](https://www.anaconda.com/products/distribution)
2. [CUDA 11.3 or later](https://developer.nvidia.com/cuda-downloads)
3. [PyTorch 1.12 or later](https://pytorch.org/get-started/locally/)
4. [Jupyter Notebook](https://jupyter.org/)

---
