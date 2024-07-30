#!/bin/bash
pip install gdown
pip install -U datasets
pip install -U transformers
pip install seqeval==1.2.2
pip install sentencepiece
pip install timm==0.4.12
pip install Pillow
pip install einops
pip install textdistance
pip install shapely
pip install numpy
pip install accelerate -U
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone https://github.com/microsoft/unilm.git
pip install -e unilm/layoutlmv3
