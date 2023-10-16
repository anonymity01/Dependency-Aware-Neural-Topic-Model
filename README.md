# Dependency-Aware Neural Topic Model.

Code for the paper: [Dependency-Aware Neural Topic Model](https://doi.org/10.1016/j.ipm.2023.103530) (Information Processing and Management, 2024), by Heyan Huang, Yi-Kun Tang, Xuewen Shi and Xian-Ling Mao.

# Requirement
- python 3.6.5
- torch 1.7.1

# Data format
- Follow data in paper: [Neural Variational Inference for Text Processing (2016)](https://github.com/ysmiao/nvdm/tree/master/data/20news)

- id_label_50.txt.train (test): each line represents the labels of each document, split by space

- label_fre_sorted_list.txt: each line represents a name of the label

# Run:
- CUDA_VISIBLE_DEVICES=0 python ./run_depntm.py --hidden 512 --topicembedsize 256 --topics 50  --data-path data_path --vocab-size 10050  --label-path id_label_50.txt.train --label-voc-path label_fre_sorted_list.txt --fname abstract_50 --gama 64 --z2-dim 64 --model-save-path output_dir --background-topics 5

If you find this code useful, please cite our paper:
@article{HUANG2024103530,
title = {Dependency-Aware Neural Topic Model},
journal = {Information Processing & Management},
volume = {61},
number = {1},
pages = {103530},
year = {2024},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103530},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323002674},
author = {Heyan Huang and Yi-Kun Tang and Xuewen Shi and Xian-Ling Mao},
}