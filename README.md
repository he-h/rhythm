

<div align="center">


# [NeurIPS '25] RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility

[![arXiv](https://img.shields.io/badge/arXiv-RHYTHM-ff0000.svg?style=for-the-badge)]()  [![Github](https://img.shields.io/badge/RHYTHM-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/he-h/rhythm)
</div>


This repository implements the RHYTHM framework for spatio-temporal mobility prediction using pre-computed semantic embeddings.


The repo contains: 
1. **Official Implementation**  
The official implementation of RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility.







## Prerequisites

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`


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

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- AutoTimes (https://github.com/thuml/AutoTimes)
- ST-MoE-BERT (https://github.com/he-h/ST-MoE-BERT)
- LLM-Mob (https://github.com/xlwang233/LLM-Mob)

## Citation

If you have any questions regarding our paper or code, please feel free to start an issue.

If you use RHYTHM in your work, please kindly cite our paper:

```
@article{
}
```

## Contact

If you have any questions or want to use the code, feel free to contact:
* Haoyu He (he.haoyu1@northeastern.edu)
* Robin Luo (hluo@u.northwestern.edu)
