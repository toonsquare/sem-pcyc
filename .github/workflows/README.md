# SEM PCYC CML & DVC

## github action의 runner
runner는 이용자에게 부여된 가상 머신정도로 이해  

* host runner
1. OS, 사전에 설치된 Package 및 도구, self hosted runner 애플리케이션에 대한 자동 업테이트를 진행
2. github에서 관리 및 유지  
3. 모든 작업 실행에 대해 깨끗한 인스턴스를 제공함

* self hosted runner
1. 이미 비용을 지불한 클라우드 서비스 또는 로컬 머신을 사용할 수 있음  
2. 모든 작업 실행에 대해 깨끗한 인스턴스가 필요하지 않음  
3. github action과 함께 무료로 사용할 수 있지만, runner machine을 유지, 관리하는 비용은 사용자가 부담해야 한다.

sem-pcyc 모델은 nipa 서버에서 작업을 수행하기 때문에 github의 host runner대신 self hosted runner를 사용하였음  
host runner를 사용할 경우 nipa에 출발지 ip를 등록해주어야 하기 때문에, nipa 서버를 self hosted ruuner로 사용하였음  

self hosted runner docs  
https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#ip-addresses-of-github-hosted-runners  
https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners  
https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners  

## github action - self hosted runner 등록 및 사용
Setting 탭 -> Action 탭 -> Runners -> New self-hosted runner 선택

1. 운영체제 선택 (window / linux / macOS)
2. 사용하고자 하는 local machine 또는 가상 서버의 terminal로 들어가서 Download의 script 입력
3. 그 아래 Configure의 script 입력  
(저는 machine name을 제외한 나머지 configure는 defalt로 설정하였습니다. 그리고 label은 yml파일을 작성할 때 runs-on에서 runner를 선택할 때 사용됩니다.)
4. 생성된 action-runner 폴더로 이동한 후 ./run.sh를 입력하여 github action과 runner를 연결한다.

## .yml 파일 작성
* **workflow**  
**하나의 event가 trigger되었을 때, runner가 job에 있는 여러 step들의 command를 수행하는 일련의 흐름정도로 이해**

* **on**  
trigger가 되기 위한 event를 정하는 단계  
대표적으로 push와 pull request가 있으며, 수동으로 작동할 수 있게 해주는 workflow dispathch가 있음  
ML모델의 경우 코드의 수정이 많지 않기 때문에 event를 push나 pull requst로 주지 않고 workflow disptch를 사용하였다.  
다양한 event: https://docs.github.com/en/actions/learn-github-actions/events-that-trigger-workflows

* **job**  
workflow에서 노드로 표현되는 각각의 단계  
job은 `needs` 옵션을 주어 선행 job에 의존성을 부여할 수 있고 이를 통해 job들이 직렬적으로 수행하도록 할 수 있다.  
만약 선행관계가 주어지지 않았을 경우 job들은 병렬로 수행된다.

* **runs-on**  
runner를 선택하는 단계  
1. github host runner를 사용할 경우  
window, linux, macOS 중 하나를 선택할 수 있음  
`runs-on: ubnutu-latest`  

2. self hosted runner를 사용할 경우
앞서 self hosted runner를 등록할 때 정했던 label을 적어줌(label은 변경이 가능하다.)  
`runs-on: self-hosted`

* **env**  
환경 변수를 지정하는 단계  
```
env:
   FIRST_NAEM: Mona
   MIDDLE_NAME: The
   LAST_NAME: Octocat

run: echo "Hello $FIRST_NAEM $MIDDLE_NAME $LAST_NAME"
```
민감한 정보를 secret으로 암호화하여 변수로 사용하기  
Setting 탭 -> Secret 탭 -> New repository secret
현재 aws ssh에 접속하기 위한 PEM_KEY & AWS_ID & AWS_IP를 secret으로 가렸음
```
env: 
  PEM_KEY: ${{ secrets.PEM_KEY }}
  AWS_IP: ${{ secrets.AWS_IP }}
  AWS_ID: ${{ secrets.AWS_ID }}
  
run: echo aws_ip is ${{ secrets.AWS_IP }}
```
암호화된 변수를 사용하면 *** 로 가려져서 출력된다.

* **run**  
command를 실행하는 단계  
`run: | `으로 하면 여러 command를 입력해줄 수 있다.  

### 이미 github action이 connect되어 있는 경우

![git action과 연결이 끊어졌을 때](https://user-images.githubusercontent.com/82593754/145521300-d3acf0d2-e8de-4b3a-bcb7-e7159a74b4e4.png)

이러한 경우 강제로 프로세스를 종료해주어야 한다.
1. 터미널에서 `ps` 또는 `ps -ef`를 통해 프로세스의 목록을 확인
2. run 프로세스를 찾은 후 PID를 확인
3. `kill -9 PID` 명령어로 프로세스를 종료
프로세스 종료 후 바로 run이 되지 않았고 1 ~ 3분 정도 후 시도하니까 잘 연결이 되었다.

## Pipeline
![sem_pcyc_cml_dvc](https://user-images.githubusercontent.com/82593754/146878527-af0acdef-6ff8-4629-8486-9245c18726b0.jpg)

**사용자가 새로운 dataset을 dvc push와 git push를 이용해서 data registry(git repo)와 remote storage(s3 bucket)에 올려주는 것을 전제로 한다(<u>dataset을 추가할 때는 tag 달지 말것!</u>).**

### Pipeline에 대한 설명
1. <u>사용자가 클래스가 추가된 intersection dataset을 dvc push로 s3 bucket에, git push로 data registry에 올렸음을 가정함</u>
2. github action을 trigger하기 위해 workflow_dispatch를 수동으로 클릭
3. job1에서 dvc get을 이용하여 클래스가 추가된 intersection dataset을 nipa의 dataset 경로에 위치시킴
4. train을 하고 mar을 비롯한 여러 파일들(pth, npy, mar, dataset ...)과 code들에 tag를 붙이고 git push & dvc push를 해준다.
5. 클래스가 추가된 dataset으로 만들어진 mar을 request한다.
6. scp로 파일들을 aws ssh에 이동시킨다.
7. 새로운 모델을 등록한다.


Create files and train a model  
train 제외 약 3분 소요  
train도 확인 - 하나의 job은 최대 7시간 동안 작동한다는 점을 고려해야 함  

Request  
약 1분 50초 소요
sleep을 이용해서 약간의 텀을 주고 requset를 실행  
만약 sleep을 주지 않으면, torchserve start를 한 후 바로 requset를 실행하기 때문에 에러가 발생함  

Transfer files  
약 3분 소요  
dataset까지 전송을 시켜주기 때문에 나중에 더 많은 dataset이 추가된다면 소요 시간은 더 늘어날 것으로 예상  

Register a model  
약 40초 소요  

# DVC
dvc 설치  
`pip install dvc`  

NIPA의 경우 conda로 dvc 가상환경을 만들었음  
`source activate pytorch_p36` 입력 후 `conda activate DVC` 입력

## dvc로 파일 push하기
version 관리를 하고 싶은 파일이 bigdata.txt라 가정
1. `git init`
2. `dvc init`
3. `dvc add bigdata.txt`  
bigdata.txt.dvc 파일 생성, .gitignore에 bigdata.txt 파일이 자동 등록됨   
4. `git add bigdata.txt.dvc .gitignore`   
.dvc파일과 .gitignore git add 하기  
5. `dvc remote add -d 저장소_이름설정 s3://bucket/path`   
real file을 저장할 remote storage 등록(s3라 가정, 경로에는 bucket의 uri입력)  
6. `git add .dvc/config`  
remote storage를 등록하면 .dvc/config 파일에 정보가 입력되며, 이것도 git add 해주기  
7. `git commit -m "message"`  
data registry에 push하기 위해 commit을 함  
8. `git tag -a "v1.0" -m "message"`  
version관리를 하기 위해 tag를 달아줌    
9. `git push`  
지정된 data registry에 .dvc 파일 push(real file은 .gitignore에 있어 github에 올라가지 못함)    
10. `git push --tags`  
tag도 push    
11. `dvc push`  
지정된 remote storage에 real file push  
    (s3를 remote storage로 지정한 경우, `pip install boto3` & `pip install s3fs`를 해주어야 한다.)

## dvc로 파일 pull하기
현재 remote storage가 s3이므로 dvc remote modify로 aws access key와 id를 적어주어야 s3에 접근이 가능하다.  
따라서 file을 가져오기 위해서는 <u>클론을 먼저 한 후 key & id를 등록을 해주고 file을 가져와야 한다.</u>  
**(github action이라면 환경변수 등록해주었기 때문에 어떤 경로에서든지 file을 가져올 수 있다.)**

### git clone 해서 file가져오기(_dvc pull_)
1. `git clone https://github.com/toonsquare/sem-pcyc.git`
2. `git tag`  
클론한 repository에 tag의 목록을 보여줌
3. `git checkout 버전`
4. `dvc pull`  
해당 버전에서 push 했던 file들을 가져옴


### data registry(git repository)에서 file가져오기(_dvc get & dvc import_)
* intersection.tar 파일 가져오기      
`dvc get https://github.com/toonsquare/sem-pcyc.git DVC/intersection.tar`  
`dvc get` 명령어만 동작이 안되고 있으며, github action에서만 동작이 된다.   
오류는 `ERROR: unexpected error - Unable to locate credentials ` 이며, aws access key와 id를 등록하지 않고 dvc pull이나 push를 할 때 발생한다.  
그러나 key와 id를 등록을 했음에도 발생하고 있다.    


* 파일이나 폴더 가져오기 (DVC.dvc와 DVC폴더 생성)  
`dvc import https://github.com/toonsquare/sem-pcyc.git DVC`

  
## DVC pull & push할 때 주의사항
dvc push나 pull을 할 때, 지정된 remote storage에 맞는 credential 정보를 입력해주어야 한다.  
ex) s3 bucket - access_key_id & secret_access_key를 local에 등록을 해준다.  
(민감한 정보는 --local로 config.local에 등록해줄 수 있다.)  
`dvc remote modify --local storage access_key_id 'my id'`  
`dvc remote modify --local storage secret_access_key 'my key'`  
이렇게 지정을 해주면 key와 id가 .dvc/config.local 파일에 등록되고, 이 파일은 .gitignore에 있어 push가 되지 않는다.  

github action으로 push나 pull을 할 경우에는 `dvc remote modify` 명령어를 사용할 필요 없이,  
`- uses: iterative/setup-dvc@v1secret`를 사용하고, key와 id를 환경변수로 지정해 주면 dvc를 자동화하여 사용할 수 있다.
```
steps:
    - uses: actions/checkout@v2
    - uses: iterative/setup-dvc@v1
       env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```




