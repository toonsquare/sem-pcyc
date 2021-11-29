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

*

```bash
conda install _pytorch_select 
```

## 가상환경 생성
- 참고 https://varhowto.com/install-pytorch-1-4-0/
```bash
conda create --name 가상환경이름(ex.torch)
conda activate 가상환경이름(ex.torch)
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```
```
 conda list | grep torch
pytorch                   1.4.0           py3.6_cuda10.1.243_cudnn7.6.3_0    pytorch
torchvision               0.2.2.post3              pypi_0    pypi
```
* 필요 패키지 설치

- torch
- torchvision
- numpy
- scikit-learn
- google ( 데이터를 온라인 다운로드 받을 경우 사용, 로컬에 저장해서 사용할때는 필요 없음 )
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

Multi gpu 중 지정해서 테스트

```
CUDA_VISIBLE_DEVICES=0,1 python3 src/test.py --dataset TU-Berlin --dim-out 64 --semantic-models hieremb-path word2vec-google-news --batch-size 64 --save-image-results
```

### Train

##### TU-Berlin

```bash
python3 src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001
```

- train 시 CUDA memory에 따라 batch size 변경 필요
- CUDA_VISIBLE_DEVICES=1 뜻은 GPU 0 번(1080 Super), 1 번(2080TI) 사용시 1번 사용고 싶을 때 사
![test image](./figures/multi_gpu.png)

example.

```bash
python3 src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001 --batch-size 64
```

Multi gpu 중 지정해서 학습 시키기

```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 1000 --early-stop 200 --lr 0.0001 --ngpu 2 --batch-size 64
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

### Train 시 저장될 폴더

path_sketch_model

```
/ml_data/sem_pcyc/aux/CheckPoints/TU-Berlin/sketch
```

path_image_model

```
/ml_data/sem_pcyc/aux/CheckPoints/TU-Berlin/image
```

path_cp

```
/ml_data/sem_pcyc/aux/CheckPoints/TU-Berlin/word2vec-google-news/64
```

### NIPA
/etc/hosts 에 아래 와 같이 추가
```
49.50.175.252 nipa
```



# Start serving

## Images embedding
/src/mk_image_emb_npy.py 파일을 사용하여 images_embedding.npy 파일 생성


## Make mar file
```
torch-model-archiver --model-name sem_pcyc --version 1.0 --model-file ./src/models.py --serialized-file ./model_best.pth --extra-files ./images_embedding.npy --handler ./ts/torch_handler/sem_pcyc_handler.py
```
### Option explain
- --model-name : 모델 이름
- --model-file : 모델 아카이브할때 필요한 클래스와 함수가 저장된 파일
- --serialized-file : 모델의 가중치가 저장된 pth파일
- --extra-files : 필요한 추가적인 파일, images_embedding.npy의 파일명과 경로가 handler에서 npy가져올때 사용하는 파일명과 경로가 같게 설정해야한다
- --handler : handler file
- 추가적인 옵션은 https://github.com/pytorch/serve/blob/master/model-archiver/README.md 참고

## Torchserve start
```
torchserve --ts-config ./ts/config.properties --foreground --start --model-store ./model-store --models sem_pcyc=sem_pcyc.mar
```
옵션은 https://github.com/pytorch/serve 참고
 
## Production

### pytorch serving article
- https://pytorch.org/serve/index.html
- https://towardsdatascience.com/how-to-deploy-your-pytorch-models-with-torchserve-2452163871d3


Docker 로 torchserver 실행방법
```
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples  pytorch/torchserve:latest torchserve --start --model-store /home/model-server/model-store --models densenet161=densenet161.mar
```

## Version 관리
### Torchserve를 신규 모델 등록 및 버전 관리 
1.	신규 모델 등록
다중 모델 등록 가능, 필요에 따라 동시에 사용 할 수 있다.
-	모델 등록 시 모델 명을 따로 넣지 않을 경우 mar 파일명이 model 명으로 등록된다.
2.	버전 관리
-	버전은 mar 파일을 만들 때 파일 이름(--model-name 옵션)에 명시
-	최초 버전은 아무것도 표기하지 않는다.
Ex) 최초 버전 : sem_pcyc.mar | 그후에 추가하고 싶은 버전 : sem_pcyc_1.0.1, sem_pcyc_2.0.3
-	새로운 버전 등록 시 꼭 모델 명을 넣어야 새로운 모델이 아닌 기존 모델에 다른 버전이 등록 된다. 모델 명을 입력하지 않을 경우 mar 파일에서 모델명을 가져오기 때문에 이전의 등록된 모델명과 달라 새로운 모델로 등록된다.
-	Mar 파일은 삭제하지 않는다.(중요!!)
3.	Dokcer shutdown 발생 시 대응책
-	만약 docker가 내려갔다 다시 켜 질 경우 가장 최근 로그(/logs/config/shutdown.cfg)를 사용하여 torchserve를 시작
```
torchserve --log-config shutdown.cfg --foreground --start --model-store ./model-store 
```

자세한 실행 코드는 아래 코드 참고

모델 archive(version 2.0)
```
torch-model-archiver --model-name sem_pcyc --version 2.0 --model-file ./src/models.py --serialized-file ./model_best.pth --handler ./ts/torch_handler/sem_pcyc_handler.py
```
model register
```
curl -X POST "http://localhost:8081/models?url=/model-store/sem_pcyc.mar"

model version 2.0 등록
curl -X POST "http://localhost:8081/models?model_name=sem_pcyc&url=/home/ubuntu/sem-pcyc/model-store/sem_pcyc2.0.mar"
```
모델 확인
```
curl http://localhost:8081/models/sem_pcyc/2.0
```
default 모델 변경(version 2.0을 default 모델로 변경)
```
curl -v -X PUT http://localhost:8081/models/sem_pcyc/2.0/set-default
```
모델 2.0 version 확인
```
curl http://localhost:8081/models/sem_pcyc
```
모델 등록 취소(version 1.0을 등록 취소할 경우)
```
curl -X DELETE http://localhost:8081/models/sem_pcyc/
```
Request 하기
```
curl http://0.0.0.0:8080/predictions/sem_pcyc -T guitar.png
```


log사용하여 시작하기

torchserve --start를 했을때 logs/config 디렉토리가 생긴다. 여기에 가장 최근 로그를 불러올 수 있다.(단, 관련 mar 파일들이 모두 model-store에 있어야한다.)
```
torchserve  --foreground --start --model-store ./model-store --log-config ./logs/config/20211028083058376-shutdown.cfg
```

## class 추가 후 mar version update하기
### class 추가시 word embedding하기
### word2vec.npy 생성하기

data set에 있는 class가 추가되었을 때, 그 class에 맞는 embedding들이 필요하다.  
src/newclass_word2vec.py에 있는 create_wordemb()함수를 사용하면,  
추가된 클래스에 대한 word embedding된 값들을 반환한다.

### 모델 train하기
이렇게 나온 word2vec.npy는 train시 사용되며, 해당 semantic 모델명으로 pth가 담긴 폴더가 생성된다.
```
python train.py --dataset intersection --dim-out 64 --semantic-models new_plus_words --epochs 1000 --early-stop 200 --lr 0.0001 --batch-size 32 --save-image-results
```

### .mar 생성하기
mar 파일을 생성하기 위해서는 2가지가 필요한데, model_best.pth와 acc_im_em.npy이다.  
acc_im_em.npy는 src/mk_image_emd.py를 통해 만들 수 있다.  
이 파일에서도 semantic_models = []에 word2vec npy의 이름을 적어주고,
```
    def _load_files_tuberlin_zeroshot( self,root_path, split_eccv_2018=False, photo_dir='images', sketch_dir='sketches',
                                      photo_sd='', sketch_sd='', dataset=''):
        path_im = os.path.join(root_path, photo_dir, photo_sd)
        path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

        # image files and classes
        if dataset == '':
            fls_im = glob.glob(os.path.join(path_im, '*', '*'))
        else:
            fls_im = glob.glob(os.path.join(path_im, '*', '*.base64'))
```
fls_im가 base64가 출력이 되도록 해준다. 만약 base64가 출력되는 것이 아니라 aug 파일명까지 출력된다면 if문 전체를 base64로 맞춰준다.

sem_pcyc_handler의 경우, aws ssh에 mar, npy, pth, 추가된 dataset 등의 파일을 전송해야 하므로 그와 같은 경로를 맞춰줄 필요가 있다.  
```
self.npy_path = '/home/model-server/npy'
path_dataset = '/home/model-server/sem_pcyc/dataset'
path_aux = '/home/model-server/sem_pcyc/aux'
semantic_models = ['new_plus_words']
```

```
    def _load_files_tuberlin_zeroshot( self,root_path, split_eccv_2018=False, photo_dir='images', sketch_dir='sketches',
                                      photo_sd='', sketch_sd='', dataset=''):
        path_im = os.path.join(root_path, photo_dir, photo_sd)
        path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

        # image files and classes
        if dataset == '':
            fls_im = glob.glob(os.path.join(path_im, '*', '*.base64'))
        else:
            fls_im = glob.glob(os.path.join(path_im, '*', '*.base64'))
```
로 변경해주어야 한다. handler의 경우 else문만 base64로 바꾸었더니 제대로 파일을 분류하지 못하는 문제가 발생하였다.

만약 fls_im의 size가 다르다면, 성능이 매우 낮아지므로 acc_im_em과 handler에서의 fls_im size를 같게 맞춰주는 것이 중요하다.

archiving을 하여 mar파일을 생성한다.  
```
torch-model-archiver --model-name sem_pcyc2.0 --version 2.0 --model-file ./src/models.py --serialized-file ./src/model_best.pth --handler ./ts/torch_handler/sem_pcyc_handler.py --extra-files ./npy/acc_im_em.npy
```

### aws ssh에 접속하여 model-store에 mar파일 올리기
scp 를 이용하여 파일 및 디렉토리를 전송할 수 있다.  
pem이 키이며, 해당 키가 있는 경로에서 명령어를 사용해야 permission denied 오류가 발생하지 않는다.  
```
예) .mar 전송
scp -i ./pem키 /home/ubuntu/projects_jonathan/model-store/sem_pcyc2.0.mar ubuntu@13.209.76.135:/home/ubuntu/sem-pcyc/model-store

예) .images dataset 전송
scp -r -i ./pem키 /home/ubuntu/sem_pcyc/dataset/intersection/images/ ubuntu@13.209.76.135:/home/ubuntu/ml_data/sem_pcyc/dataset/intersection/images
```

aws ssh에 mar, npy(acc_im_em & semantic), pth, dataset 등을 경로에 맞게 전송을 해야 한다.  
npy 파일(semantic_model)의 경로는 home/ubuntu/ml_data/sem_pcyc/aux/Semantic/intersection
pth 파일의 경로는 home/ubuntu/ml_data_sem_pcyc/aux/CheckPoints/intersection/모델명/64/model_best.pth
npy 파일(acc_im_em.npy)의 경로는 home/ubuntu/ml_data/sem_pcyc/npy
mar 파일의 경로는 home/ubuntu/sem-pcyc/model-store  
dataset의 경로는 home/ubuntu/ml_data/sem_pcyc/dataset/intersection/images 또는 sketches

### version update하기
파일들이 경로에 맞게 위치해있다면, model 등록할 수 있다.  
```
curl -X POST "http://localhost:8081/models?model_name=sem_pcyc&url=/home/ubuntu/sem-pcyc/model-store/sem_pcyc2.0.mar"
```
을 하면 "status": "Model /"sem_pcyc/" Version: 2.0 registered with 1 initial workers" 라는 메시지와 함께 등록이 된다.  
만약, 오류가 발생한다면,(특히 "code":500) 

```
docker logs mar
```
로그를 통해 오류를 확인할 수 있다.  

등록이 잘 되었는지 model을 확인할 수 있다.  
```
curl http://localhost:8081/models/sem_pcyc/2.0
```

등록이 잘 되었다면 version 2를 default모델로 지정해준다.  
```
curl -v -X PUT http://localhost:8081/models/sem_pcyc2.0/set-default
```


### reference docs
- https://pytorch.org/serve/management_api.html
- https://pytorch.org/serve/server.html

## Author

* [Anjan Dutta](https://sites.google.com/site/2adutta/) ([@AnjanDutta](https://github.com/AnjanDutta))


### Reference
https://neptune.ai/blog/6-gan-architectures
https://eehoeskrap.tistory.com/430