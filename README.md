<div align="center">


# RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility

[![arXiv](https://img.shields.io/badge/arXiv-RHYTHM-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2509.23115)  [![Github](https://img.shields.io/badge/RHYTHM-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/he-h/rhythm)
</div>




The official implementation of [**RHYTHM**: Reasoning with Hierarchical Temporal Tokenization for Human Mobility](https://arxiv.org/abs/2509.23115).

## Contents

- [**RHYTHM**: Reasoning with Hierarchical Temporal Tokenization for Human Mobility](https://arxiv.org/abs/2509.23115) 
  - [1. Introduction](#introduction)
  - [2. Prerequisites](#prerequisites)
  - [3. Data](#data)
  - [4. Usage](#usage)
    - [4.1 Preprocess](#1-preprocess)
    - [4.2 Training](#2-training)
    - [4.3 Evaluate](#3-evaluate)
  - [5. Acknowledgement](#acknowledgement)
  - [6. Citation](#citation)
  - [7. Contact](#contact)
 


## Introduction
**RHYTHM** (Reasoning with Hierarchical Temporal Tokenization for Human Mobility) reframes mobility prediction through a foundation-model lens: it compresses long trajectories into structured temporal tokens, uses hierarchical attention to capture daily/weekly rhythms, and injects pre-computed prompt-guided semantic context—all while keeping the LLM backbone frozen for lightweight, scalable adaptation. The result is a simple, compute-efficient recipe that preserves LLM reasoning, travels well across cities, and hints at practical scaling behavior for spatio-temporal foundation models.






## Prerequisites

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

```
# create and activate virtual python environment
conda create -n rhythm python=3.10
conda activate rhythm
pip install transformers
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# install required packages
pip install -r requirements.txt
```

## Data

Our experiments are conducted on the [YJMob100k](https://www.nature.com/articles/s43588-024-00650-3) dataset.

## Usage



### 1. Preprocess

First, preprocess your raw trajectory files and compute semantic embeddings offline:

```bash
# Preprocess raw data
bash scripts/preprocess.sh
```

### 2. Training
Trained model outputs, logs, and final metrics are saved under the log/ directory, and checkpoints are saved under the checkpoints/ directory.
You can train the model on different cities by running the following scripts:

```bash
# Train on City B
bash scripts/b.sh

# Train on City C
bash scripts/c.sh

# Train on City D
bash scripts/d.sh
```

### 3. Evaluate
Coming soon...

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- AutoTimes (https://github.com/thuml/AutoTimes)
- ST-MoE-BERT (https://github.com/he-h/ST-MoE-BERT)



## Citation

If you have any questions regarding our paper or code, please feel free to start an issue.

If you use RHYTHM in your work, please kindly cite our paper:

```
@misc{he2025rhythmreasoninghierarchicaltemporal,
      title={RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility}, 
      author={Haoyu He and Haozheng Luo and Yan Chen and Qi R. Wang},
      year={2025},
      eprint={2509.23115},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23115}, 
}
```

## Contact

If you have any questions or want to use the code, feel free to contact:
* Haoyu He (he.haoyu1@northeastern.edu)
* Robin Luo (hluo@u.northwestern.edu)
