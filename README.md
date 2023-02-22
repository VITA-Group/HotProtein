# HotProtein: A Novel Framework for Protein Thermostability Prediction and Editing
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of ICLR 2023 paper [HotProtein: A Novel Framework for Protein Thermostability Prediction and Editing](https://openreview.net/forum?id=YDJRFWBMNby&noteId=vl18g7_lVT).
## Abstract
The molecular basis of protein thermal stability is only partially understood and has major significance for drug and vaccine discovery.  The lack of datasets and standardized benchmarks considerably limits learning-based discovery methods. We present HotProtein, a large-scale protein dataset with growth temperature annotations of thermostability, containing 182K amino acid sequences and 3K folded structures from 230 different species with a wide temperature range $-20^{\circ}\texttt{C}\sim 120^{\circ}\texttt{C}$. Due to functional domain differences and data scarcity within each species, existing methods fail to generalize well on our dataset. We address this problem through a novel learning framework, consisting of (1) Protein structure-aware pre-training (SAP) which leverages 3D information to enhance sequence-based pre-training; (2) Factorized sparse tuning (FST) that utilizes low-rank and sparse priors as an implicit regularization, together with feature augmentations. Extensive empirical studies demonstrate that our framework improves thermostability prediction compared to other deep learning models. Finally, we introduce a novel editing algorithm to efficiently generate positive amino acid mutations that improve thermostability.
## Usage

### Environment

```bash
pip install -e .
pip install wandb
pip install pytorch
```
###  Datasets

HP-S2C2: [Google Drive](https://drive.google.com/file/d/1Mn07gsZAfSK4YiP2oXZe1Q6PD4t6t8GG/view?usp=sharing)

HP-S2C5: [Google Drive](https://drive.google.com/file/d/1-HGaVectM15hIKopCTM-PhPz9AiN7NlQ/view?usp=share_link)

HP-S: [Google Drive](https://drive.google.com/file/d/1Vi11UTWHz0JdtDc8aH78YlMZpmWh0IR5/view?usp=share_link)

### Checkpoints

Please find the outcomes of protein structure-aware pre-training (SAP) in this [link](https://drive.google.com/file/d/17KBj0QayIDGRRfWc8O-SrAedaSPY4CCP/view?usp=share_link).

### Training 

We provide sample training scripts in the `scripts` folder. 


## Acknowledgement

Our codes are developed based on [esm](https://github.com/facebookresearch/esm). 