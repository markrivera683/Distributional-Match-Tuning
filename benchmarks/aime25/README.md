---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: test
    num_examples: 30
configs:
- config_name: default
  data_files:
  - split: test
    path: test.jsonl
license: apache-2.0
---

# AIME 25

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0) 
[![AIME25 Dataset](https://img.shields.io/badge/Huggingface-Datasets-blue)](https://huggingface.co/datasets/math-ai/aime25) 

### American Invitational Mathematics Examination (AIME) 2025 

## Citation
If you use the AIME25 dataset in your research, please consider citing it as follows:

```
@misc{aime25,
      title={American Invitational Mathematics Examination (AIME) 2025}, 
      author={Zhang, Yifan and Math-AI, Team},
      year={2025},
}
```