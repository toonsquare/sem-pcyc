"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import importlib.util
import time
import torch
import numpy as np
import glob
import itertools
from scipy.spatial.distance import cdist
import random
import importlib.util
import inspect
from PIL import Image

# pytorch, torch vision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.image_emd = self.np_load(
            "./images_embedding.npy"
        )
    def np_load(self, npy_path):
        images_emd = np.load(npy_path)
        return images_emd

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties

        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )

        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        # 나중에 사용될 acc_im_em이 담긴 .npy파일을 extra_file로 해서 넣기!!!!!!
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        # extra_file =

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

            # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            print("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            print("Loading torchscript model")
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()
        print("Model file %s loaded successfully", model_pt_path)

        #  load the model, refer 'custom handler class' above for details

    def _list_classes_from_module(self, module, parent_class=None):
        """
        Parse user defined module to get all model service classes in it.
        :param module:
        :param parent_class:
        :return: List of model service class definitions
        """

        # Parsing the module to get all defined classes
        classes = [
            cls[1]
            for cls in inspect.getmembers(
                module,
                lambda member: inspect.isclass(member)
                               and member.__module__ == module.__name__,
            )
        ]
        # filter classes that is subclass of parent_class
        if parent_class is not None:
            return [c for c in classes if issubclass(c, parent_class)]

        return classes

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        print("model_def_path {}".format(model_def_path))
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        classes = [
            cls[1]
            for cls in inspect.getmembers(
                module,
                lambda member: inspect.isclass(member)
                               and member.__module__ == module.__name__,
            )
        ]
        model_class_definitions = classes
        class_size = len(model_class_definitions)

        # if len(model_class_definitions) != 1:
        #     raise ValueError(
        #         "Expected only one class as model definition. {}".format(
        #             model_class_definitions
        #         )
        #     )

        model_class = model_class_definitions[class_size - 2]
        print("class : {}".format(model_class))
        print("model_pt_path {}".format(model_def_path))

        checkpoint = torch.load(model_pt_path)
        best_map = checkpoint['best_map']
        state_dict = checkpoint['state_dict']
        sem_dim = 0
        path_dataset = '/home/ubuntu/sem_pcyc/dataset'
        path_aux = '/home/ubuntu/sem_pcyc/aux'
        dataset = 'intersection'
        semantic_models = ['word2vec-google-news']
        files_semantic_labels = []
        dim_out = 64
        str_aux = ''
        ds_var = None
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        im_sz = 224
        sk_sz = 224

        if '_' in dataset:
            token = dataset.split('_')
            dataset = token[0]
            ds_var = token[1]

        semantic_models = sorted(semantic_models)
        model_name = '+'.join(semantic_models)
        print('model_name : ' + model_name)
        # 데이터 셋 경로
        root_path = os.path.join(path_dataset, dataset)
        # 스케치 모델 경로
        path_sketch_model = os.path.join(path_aux, 'CheckPoints', dataset, 'sketch')
        print('path_sketch_model : ' + path_sketch_model)
        # 썸네일 모델 경로
        path_image_model = os.path.join(path_aux, 'CheckPoints', dataset, 'image')
        print('path_image_model : ' + path_image_model)

        path_cp = os.path.join(path_aux, 'CheckPoints', dataset, str_aux, model_name, str(dim_out))
        print('path_cp : ' + path_cp)

        # 시멘틱 모델 벡터 값 가져오기
        for f in semantic_models:
            fi = os.path.join(path_aux, 'Semantic', dataset, f + '.npy')
            print('fi : ' + fi)
            files_semantic_labels.append(fi)
            sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

        # Parameters for transforming the images
        transform_image = transforms.Compose([transforms.Resize((im_sz, im_sz)), transforms.ToTensor()])
        transform_sketch = transforms.Compose([transforms.Resize((sk_sz, sk_sz)), transforms.ToTensor()])

        print('Loading data ...')
        splits = self._load_files_tuberlin_zeroshot(root_path=root_path, split_eccv_2018=False,
                                                    photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
                                                    sketch_sd=sketch_sd)
        # Combine the valid and test set into test set
        splits['te_fls_sk'] = np.concatenate((splits['va_fls_sk'], splits['te_fls_sk']), axis=0)
        print('----te_fls_sk----')
        # print(splits['te_fls_sk'])
        splits['te_clss_sk'] = np.concatenate((splits['va_clss_sk'], splits['te_clss_sk']), axis=0)
        print('----te_clss_sk----')
        # print(splits['te_clss_sk'])
        splits['te_fls_im'] = np.concatenate((splits['va_fls_im'], splits['te_fls_im']), axis=0)
        print('----te_fls_im----')
        # print(splits['te_fls_im'])
        splits['te_clss_im'] = np.concatenate((splits['va_clss_im'], splits['te_clss_im']), axis=0)
        print('----te_clss_im----')
        # print(splits['te_clss_im'])

        dict_clss = self._create_dict_texts(splits['tr_clss_im'])

        params_model = dict()
        params_model['path_sketch_model'] = path_sketch_model
        params_model['path_image_model'] = path_image_model
        # Dimensions
        params_model['dim_out'] = 64
        params_model['sem_dim'] = sem_dim
        # Number of classes
        params_model['num_clss'] = len(dict_clss)
        print(('num_clss : {}'.format(params_model['num_clss'])))
        # Weight (on losses) parameters
        params_model['lambda_se'] = 10
        params_model['lambda_im'] = 10
        params_model['lambda_sk'] = 10
        params_model['lambda_gen_cyc'] = 1
        params_model['lambda_gen_adv'] = 1
        params_model['lambda_gen_cls'] = 1
        params_model['lambda_gen_reg'] = 0.1
        params_model['lambda_disc_se'] = 0.25
        params_model['lambda_disc_sk'] = 0.5
        params_model['lambda_disc_im'] = 0.5
        params_model['lambda_regular'] = 0.001
        # Optimizers' parameters
        params_model['lr'] = 0.0001
        params_model['momentum'] = 0.9
        params_model['milestones'] = []
        params_model['gamma'] = 0.1
        # Files with semantic labels
        params_model['files_semantic_labels'] = files_semantic_labels
        # Class dictionary
        params_model['dict_clss'] = dict_clss

        model = model_class(params_model)
        # Check cuda
        print('Checking cuda...', end='')
        # Check if CUDA is enabled
        if torch.cuda.is_available():
            print('*Cuda exists*...', end='')
            model = model.cuda()
        print('Done')

        model.load_state_dict(state_dict)
        return model

    def _create_dict_texts(self, texts):
        texts = sorted(list(set(texts)))
        d = {l: i for i, l in enumerate(texts)}
        return d

    def _get_coarse_grained_samples(self, classes, fls_im, fls_sk, set_type='train', filter_sketch=True):

        idx_im_ret = np.array([], dtype=np.int)
        idx_sk_ret = np.array([], dtype=np.int)
        clss_im = np.array([f.split('/')[-2] for f in fls_im])
        clss_sk = np.array([f.split('/')[-2] for f in fls_sk])
        names_sk = np.array([f.split('-')[0] for f in fls_sk])

        # print('clss_im size : {}'.format(len(clss_im)))
        #
        # print('fls_sk size : {}'.format(len(fls_sk)))
        # print('clss_sk size : {}'.format(len(clss_sk)))
        #
        # print('names_sk size : {}'.format(len(names_sk)))

        for i, c in enumerate(classes):
            idx1 = np.where(clss_im == c)[0]
            idx2 = np.where(clss_sk == c)[0]
            if set_type == 'train':
                idx_cp = list(itertools.product(idx1, idx2))
                # print('idx_cp size : {}'.format(len(idx_cp)))
                if len(idx_cp) > 100000:
                    random.seed(i)
                    idx_cp = random.sample(idx_cp, 100000)
                idx1, idx2 = zip(*idx_cp)
            else:
                # remove duplicate sketches
                if filter_sketch:
                    names_sk_tmp = names_sk[idx2]
                    idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                    idx2 = idx2[idx_tmp]
            idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
            idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)

        return idx_im_ret, idx_sk_ret

    def _load_files_tuberlin_zeroshot(self, root_path, split_eccv_2018=False, photo_dir='images', sketch_dir='sketches',
                                      photo_sd='', sketch_sd='', dataset=''):

        print('start load_files_tuberlin_zeroshot')
        path_im = os.path.join(root_path, photo_dir, photo_sd)
        path_sk = os.path.join(root_path, sketch_dir, sketch_sd)
        print('path_im : {}'.format(path_im))
        print('path_sk : {}'.format(path_sk))

        # image files and classes
        if dataset == '':
            fls_im = glob.glob(os.path.join(path_im, '*', '*'))
        else:
            fls_im = glob.glob(os.path.join(path_im, '*', '*.base64'))
        print('fls_im.size : {}'.format(len(fls_im)))

        fls_im = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im])
        clss_im = np.array([f.split('/')[-2] for f in fls_im])

        # sketch files and classes
        fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
        print('1 fls_sk.size : {}'.format(len(fls_sk)))
        fls_sk = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk])
        print('2 fls_sk.size : {}'.format(len(fls_sk)))
        clss_sk = np.array([f.split('/')[-2] for f in fls_sk])

        # all the unique classes
        classes = np.unique(clss_im)

        # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
        np.random.seed(0)
        tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False)
        va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.06 * len(classes)), replace=False)
        te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

        idx_tr_im, idx_tr_sk = self._get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train')
        idx_va_im, idx_va_sk = self._get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid')
        idx_te_im, idx_te_sk = self._get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test')

        splits = dict()

        splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
        splits['va_fls_sk'] = fls_sk[idx_va_sk]
        splits['te_fls_sk'] = fls_sk[idx_te_sk]

        splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
        splits['va_clss_sk'] = clss_sk[idx_va_sk]
        splits['te_clss_sk'] = clss_sk[idx_te_sk]

        splits['tr_fls_im'] = fls_im[idx_tr_im]
        splits['va_fls_im'] = fls_im[idx_va_im]
        splits['te_fls_im'] = fls_im[idx_te_im]

        splits['tr_clss_im'] = clss_im[idx_tr_im]
        splits['va_clss_im'] = clss_im[idx_va_im]
        splits['te_clss_im'] = clss_im[idx_te_im]

        return splits

    def sketch_preprocessing(self, sketch):
        '''
        한 장의 스케치를 유클리디안 거리를 계산하기 위한 embedding을 만드는 전처리 과정
        argument에 sketch 파일의 경로를 적어준다.
        '''
        print("-------------start sketch_preprocessing-------------")
        print("sketch name :house")
        sketch = "/home/ubuntu/projects/src/house.png"
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        # sketch는 흑백이라 1채널인데, get_sketch_embeddings()하기 위해서는 3채널로 변환해주어야 한다.
        sketch = Image.open(sketch).convert(mode="RGB").resize((224, 224))

        # sketch를 tensor화 해주어야 한다.
        sk = transform(sketch)

        # 여러 장 쌓여있는 이미지들과
        # 비교하기 위하여 3차원의 sketch를 4차원으로 늘리기
        sk = sk.unsqueeze(dim=0)
        print("sk_dim:", sk.shape)

        # 모델을 처리하는 device를 cpu와 cuda 둘 중 하나로 맞춰줘야하기 때문에 cuda()로 device를 변환해주기
        if torch.cuda.is_available():
            sk = sk.cuda()

        # sketch 한 장을 embedding하기
        sk_em = self.model.get_sketch_embeddings(sk)
        print("-------------end embeddings-------------")

        # 유클리디안 거리 유사도 계산을 하기 위해 embedding된 tensor를 numpy array로 변환해주기
        sk_em = sk_em.cpu().detach().numpy()
        print("sk_em=",sk_em)
        print("-------------END sketch_preprocessing-------------")

        return sk_em

    # def images_preprocessing(self):
    #     '''
    #     여러 장의 썸네일들을 유클리디안 거리를 계산하기 위한 embedding을 만드는 전처리 과정
    #     argument에 썸네일들의 경로를 적어준다.
    #     '''
    #     print("-------------start image_preprocessing-------------")
    #     path_dataset = '/home/ubuntu/sem_pcyc/dataset'
    #     dataset = 'intersection'
    #     root_path = os.path.join(path_dataset, dataset)
    #     photo_dir = 'images'
    #     sketch_dir = 'sketches'
    #     photo_sd = ''
    #     sketch_sd = ''
    #     image_test = self._load_files_tuberlin_zeroshot(root_path=root_path, split_eccv_2018=False,
    #                                                     photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
    #                                                     sketch_sd=sketch_sd)
    #
    #     # 클래스명이 필요하기 때문에 image_test라는 튜플의 첫 번째 인덱스를 all_clss_im으로 저장
    #     # str_sim이 필요가 없지만 DataGeneratorImage 클래스에서는 clss_im이 사용되므로 일단 구해놓기
    #     all_clss_im = image_test["tr_clss_im"]
    #     all_fls_im = image_test["tr_fls_im"]
    #
    #     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    #
    #     dataset = 'intersection'
    #     photo_dir = "images"
    #     photo_sd = ""
    #
    #     # 이미지들을 tensor화 하는 단계
    #     data_test_image = self.model.DataGeneratorImage(dataset, root_path, photo_dir, photo_sd, all_fls_im,
    #                                                     all_clss_im, transforms=transform)
    #
    #     # tensor화된 이미지들을 load하는 단계
    #     test_loader_image = DataLoader(dataset=data_test_image, batch_size=128, shuffle=False, num_workers=4,
    #                                    pin_memory=True)
    #
    #     # 데이터를 enumerate()를 통해 im(이미지)와 cls_im 변수에 각각 vertor와 클래스명들을 담아준다.
    #     for i, (im, cls_im) in enumerate(test_loader_image):
    #         # im : 각각의 이미지들에 대한 1차원 vector
    #         # cls_im : 각각의 이미지들에 대한 클래스(라벨링)
    #         if torch.cuda.is_available():
    #             im = im.cuda()
    #
    #         im_em = self.model.get_image_embeddings(im)
    #
    #         if i == 0:
    #             acc_im_em = im_em.cpu().data.numpy()
    #             acc_cls_im = cls_im
    #         else:  # acc_im_em과 im_em을 concatenate한 것인데, 이것을 cpu로 담고 numpy() 배열로 바꿔주어 추후 cdist에 사용할 수 있게된다.
    #             acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)  # 유클리디안
    #             acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)  # str_sim에 사용 -> 필요 없음
    #     print("-------------END image_preprocessing-------------")
    #
    #     return acc_im_em

    def inference(self, sk_em):
        '''
        구해놓은 sketch와 images들의 embedding된 2차원 행열들을 유클리디안 거리 유사도 계산을 통해 2차원 행렬의 값을 구함
        arguments에는 sketch & images들의 2차원 nparray가 필요하다
        '''

        # sketch 한 장과 여러 이미지가 쌓인 이미지들 간의 유클리디안 거리 계산
        print("-------------START inference-----------")
        sim_euc = np.exp(-cdist(sk_em, self.image_emd, metric='euclidean'))
        print("-------------END inference-----------")

        return sim_euc

    def postprocessing(self, sim_euc):
        '''
        도출해 낸 유클리디안 거리에서 가장 가까운 순서대로 index를 가져오며, 그 인덱스를 가진 썸네일 파일의 이름을 가져온다.
        arguments에는 도출한 유클리디안 거리와 root_path가 들어가는데 root_path는 fls_im을 생성하기 위해 필요하다.
        '''
        print("-------------START postprocessing-----------")


        # all_fls_im를 생성
        path_dataset = '/home/ubuntu/sem_pcyc/dataset'
        dataset = 'intersection'
        root_path = os.path.join(path_dataset, dataset)
        path_im = os.path.join(root_path, "images", "")
        all_fls_im = glob.glob(os.path.join(path_im, "*", "*.base64"))
        all_fls_im = np.array(
            [os.path.join(f.split("/")[-2], f.split("/")[-1]) for f in all_fls_im]
        )

        output = []
        # 스케치 한 장에 대한 이미지들의 유클리디안 거리 값들을 np.argsort를 통해 가까운 인덱스를 100개 가져온다.
        # np.argsort()에서 -를 붙일지 말지 정해야 함
        ind_im = np.argsort(-sim_euc[0])[:10]
        print("index(-sort):", ind_im)

        # for문을 이용하여 fls_im array에서 인덱스 값들을 추려서 output 변수에 이름들을 넣어준다.
        for i in ind_im:
            output.append(all_fls_im[i])
        print("-------------END postprocessing-----------")

        return output

    def handle(self, sketch, context):


        start_time = time.time()
        self.context = context
        metrics = self.context.metrics
        sketch_em = self.sketch_preprocessing(sketch)
        #print("root_path", root_path)



        if not self._is_explain():
            sim_euc = self.inference(sketch_em)
            output = self.postprocessing(sim_euc)
            print("inference :{}".format(output))
        else:
            #output = self.explain_handle(data_preprocess, data)
            print("inference failed !!!")
        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output

    def explain_handle(self, data_preprocess, raw_data):
        """Captum explanations handler

        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request

        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = None
        inputs = None
        target = 0

        logger.info("Calculating Explanations")
        row = raw_data[0]
        if isinstance(row, dict):
            logger.info("Getting data and target")
            inputs = row.get("data") or row.get("body")
            target = row.get("target")
            if not target:
                target = 0

        output_explain = self.get_insights(data_preprocess, inputs, target)
        return output_explain

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                self.explain = True
                return True
        return False
