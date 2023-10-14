# Dependency-Aware Neural Topic Model.

Code for the paper: [Dependency-Aware Neural Topic Model]() (Information Processing and Management, 2023), by Heyan Huang, Yi-Kun Tang, Xuewen Shi and Xian-Ling Mao.

# Requirement
- python 3.6.5
- torch 1.7.1

# data format
- Follow data in paper: [Neural Variational Inference for Text Processing (2016)](https://github.com/ysmiao/nvdm/tree/master/data/20news)

- id_label_50.txt.train (test): 

- label_fre_sorted_list.txt: 

# Run:
- CUDA_VISIBLE_DEVICES=0 python ./run_depntm.py --hidden 512 --topicembedsize 256 --topics 50  --data-path data_path --vocab-size 10050  --label-path id_label_50.txt.train --label-voc-path label_fre_sorted_list.txt --fname abstract_50 --gama 64 --z2-dim 64 --model-save-path output_dir --background-topics 5

If you find this code useful, please cite our paper.