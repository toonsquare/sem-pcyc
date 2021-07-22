# SEM-PCYC

[PyTorch](https://pytorch.org/) | [Arxiv](https://arxiv.org/abs/1903.03372)


PyTorch implementation of our SEM-PCYC model for zero-shot sketch-based image retrieval:  
[Semantically Tied Paired Cycle Consistency for Zero-Shot Sketch-based Image Retrieval](https://arxiv.org/abs/1903.03372)  
[Anjan Dutta](https://sites.google.com/site/2adutta/), [Zeynep Akata](https://ivi.fnwi.uva.nl/uvaboschdeltalab/people/zeynep-akata/)  
[CVPR, 2019](http://cvpr2019.thecvf.com/)
## 개발환경 정리
|Train|Test|Ubuntu|  GPU      |  Driver |  Cuda |Conda |Python|Pytorch|torchvision|
|----|----|------|--------   |---------|-------|------|------|-------|-----------|
|성공 |성공 |18.04 | 1080 Ti   |470.57.02|11.4   |4.7.12|3.6.13|1.4.0  py3.6_cuda9.2.148_cuddn7.6.3_0|0.5.0| 
|성공 |성공 |18.04 | 2070 SUPER|470.57.02|11.4   |4.7.12|3.6.13|1.4.0 py3.6_cuda9.2.148_cuddn7.6.3_0|0.5.0|
|성공 |성공 |20.04 | 1660 SUPER|470.57.02|11.4   |4.7.12|3.6.10|1.7.1 py3.6_cuda9.2.148_cuddn7.6.3_0|0.8.2|


## Prerequisites

+ Ubuntu
  * Ubuntu 18.04 (Tested) 
  * Ubuntu 20.04 (Tested)
+ NVIDIA GPU 
  * NVIDIA GeForce GTX 1080 Ti (Tested)
  * NVIDIA GeForce RTX 2070 SUPER (Tested)
+ CUDA Version
  * 11.4 (Ubuntu Nvidia 독점 Driver Version: 470.57.02 (Tested))
+ 7z 
+ conda install (ref : https://dambi-ml.tistory.com/6)
```bash
sudo apt-get install p7zip-full
```
* rename package 설치
```bash
sudo apt-get install -y rename
```

[comment]: <> (```bash)

[comment]: <> (conda install _pytorch_select )

[comment]: <> (```)

## 가상환경 생성
```bash
conda create --name 가상환경이름(ex.torch)
conda activate 가상환경이름(ex.torch)
```
* 필요 패키지 설치
- torch
- torchvision
- numpy
- scikit-learn
- google
- tqdm
```bash
pip install --upgrade google-api-python-client
```

- pytorch 설치 (ref :https://varhowto.com/install-pytorch-cuda-9-2/)
```bash 
conda uninstall pytorch torchvision cudatoolkit=11.0 -c pytorch
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```


## Getting Started

### Installation
* Clone this repository
```bash
git clone https://github.com/AnjanDutta/sem-pcyc.git
cd sem-pcyc

```
* Update [config.ini](https://github.com/AnjanDutta/sem-pcyc/blob/master/config.ini) (see example)
```bash
[<host>]
path_dataset = <where all the datasets will be downloaded>
path_aux = <where all the auxiliary folders and files will be saved>
```
### Download datasets

* TU-Berlin
```bash
#download_datasets.sh
Sketcy dataset 부분은 주석처리
```

```bash
bash download_datasets.sh
```
OR
```bash
# code 실행 불가 경우
나스 toonsquare/개발관련/deeptoon/sem-pcyc/data/newnew 에 TU-Berlin.zip 가져오기 
```
### Download pretrained models
* Sketchy
    * sketch
    * image
    * hieremb-jcn + word2vec-google-news
* TU-Berlin
    * sketch
    * image
    * hieremb-path + word2vec-google-news
```bash
bash download_models.sh
```
OR
```bash
# code 실행 불가 경우
나스 toonsquare/개발관련/deeptoon/sem-pcyc/pretrained-model에서 가져오기 
```
### Test

##### TU-Berlin
```bash
python3 src/test.py --dataset TU-Berlin --dim-out 64 --semantic-models hieremb-path word2vec-google-news
```
- hieremb-path-word2vec-google-news : Checkpoints폴더 내에 존재하는 pretrained model
- semantic-models 뒤 옵션 수정해서 직접 train한 모델 test 가능
- test 시 CUDA memory에 따라 batch size 변경 필요 

example.
```bash
python3 src/test.py --dataset TU-Berlin --dim-out 64 --semantic-models hieremb-path word2vec-google-news --batch-size 64
```
- test 결과 이미지로 확인하기 위해
- sem-pcyc/src/utils.py에 맨 윗줄에 아래코드 추가
```bash
from PIL import Image
```
- save-image-results 옵션 활용

example.
```bash
python3 src/test.py --dataset TU-Berlin --dim-out 64 --semantic-models hieremb-path word2vec-google-news --batch-size 64 --save-image-results
```
### Train

##### TU-Berlin
```bash
python3 src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001
CUDA_VISIBLE_DEVICES=1 python3 src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001 --batch-size 128
```
- train 시 CUDA memory에 따라 batch size 변경 필요
- CUDA_VISIBLE_DEVICES=1 뜻은 GPU 0 번(1080 Super), 1 번(2080TI) 사용시 1번 사용고 싶을 때 사
![test image](./figures/multi_gpu.png)

example.
```bash
python3 src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001 --batch-size 64
```

### Citation
```
@inproceedings{Dutta2019SEMPCYC,
author = {Anjan Dutta and Zeynep Akata},
title = {Semantically Tied Paired Cycle Consistency for Zero-Shot Sketch-based Image Retrieval},
booktitle = {CVPR},
year = {2019}
}
```

## Author
* [Anjan Dutta](https://sites.google.com/site/2adutta/) ([@AnjanDutta](https://github.com/AnjanDutta))
