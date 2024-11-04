## Miniconda3

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

mkdir -p ~/miniconda3
bash ~/Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
rm ~/Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
```

### python 环境

```bash

git clone https://github.com/king-jingxiang/YiJian-Community

conda create -n yijian python=3.10
conda activate yijian


pip install torch==2.4.0 \
    diffusers==0.30.3 \
    transformers==4.43.2 \
    vllm==0.6.1 \
    vllm-flash-attn==2.6.1

pip install yijian-community
```