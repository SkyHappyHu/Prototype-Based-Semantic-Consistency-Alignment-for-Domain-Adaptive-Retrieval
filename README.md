# PSCA (Accepted by AAAI-2026!)

This repository contains the extended paper and the official MATLAB implementation for **"Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval"**.

## 🚀 Overview

This paper provides a prototype-based alignment strategy to enhance semantic consistency in domain adaptive retrieval tasks. 

- **Main Function:** `PSCA.m` (The core implementation of the proposed algorithm).
- **Execution Entry:** `demo.m` (The main script to run the code and adjust experimental parameters).

## 📂 Project Structure

```text
.
├── PSCA.m                 # Main model function and algorithm logic
├── demo.m                 # Entry script for demonstration and testing
├── Extended_Paper.pdf     # Extended version of PSCA
├── data/                  # Dataset directory
│   └── COIL_1/            # Partial datasets (included due to size limits)
│   └── MNIST_vs_USPS/        
└── utils                    # Other auxiliary MATLAB functions and subfolders
```
## 📊 Datasets
Due to GitHub's file size restrictions, we only provide the MNIST-USPS and COIL20 datasets within this repository.

For the other datasets evaluated in the paper (e.g., Office-31, Office-Home), please refer to the official public sources. These are all publicly available datasets and can be seamlessly integrated into the data/ folder for full-scale testing.

## 🛠 Usage
1. Clone or download this repository to your local machine.
2. Open MATLAB and navigate to the project root directory.
3. Add the folders and subfolders to your MATLAB path by running:
4. Run the demonstration script:`run('demo.m');`

## 📝 Citation
If this code or the content of the paper is helpful for your research, please cite our work:
```bibtex
@inproceedings{hu2026prototype,
  title={Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval},
  author={Hu, Tianle and Lv, Weijun and Han, Na and Fang, Xiaozhao and Wen, Jie and Li, Jiaxing and Zhou, Guoxu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={26},
  pages={21867--21875},
  year={2026}
}
