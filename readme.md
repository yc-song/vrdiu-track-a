# vrdiu-track-a

This repository is for the IJCAI 2024 VRDIU (Visually Rich Form Document Intelligence and Understanding) challenge, sponsored by Google Research. We achieved 2nd place in Track A (Key Information Extraction) of the competition.

### Environment Setting

Create conda environment
```
conda create -n vrdiu python=3.8
```
Install packages
```
bash env.sh
```
Download Form-NLU[1] datasets
```
bash dataset.sh
```

### Training
```
bash train.sh
```

### Inference
Download checkpoint
```
gdown 1frEB4gJEBgdBLypVZXoXNAvlY-FX332A
unzip checkpoint.zip -d checkpoint
```
Inference
```
bash inference.sh
```
Post-processing for ensuring no overlapped keys and save it to csv file
```
python post_process.py
python submission.py
```

[1] Ding, Y., Long, S., Huang, J., Ren, K., Luo, X., Chung, H., & Han, S. C. (2023, July). Form-NLU: Dataset for the Form Natural Language Understanding. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2807-2816).
