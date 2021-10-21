# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import inspect
import torch
import os
import importlib.util
import glob
import logging
import numpy as np
import itertools
import torchvision.transforms as transforms
import random
from io import BytesIO
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.im_sz = None
        self.sk_sz = None
        self.transform_sketch = None
        self.dataset = None
        self.num_workers = 4
        self.batch_size = 32
        self.test_loader_image = None
        self.root_path = None
        self.photo_dir = None
        self.photo_sd = ''
        self.sketch_sd = ''
        self.sketch_dir = None
        self.splits_test =[]
        self.splits_train =[]
        self.image_emd = self.np_load(
            "./images_embedding.npy"
        )
    # embedding npy file load
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
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

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

        # self.model.to(self.device)
        # self.model.eval()
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

        print(model_class_definitions)
        for cls in model_class_definitions:
            print(cls)
        model_class = model_class_definitions[class_size - 2]
        data_generator_image = model_class_definitions[1]
        print("class : {}".format(model_class))
        print("model_pt_path {}".format(model_def_path))

        checkpoint = torch.load(model_pt_path)
        best_map = checkpoint['best_map']
        state_dict = checkpoint['state_dict']
        sem_dim = 0
        path_dataset = '/home/ubuntu/sem_pcyc/dataset'
        path_aux = '/home/ubuntu/sem_pcyc/aux'
        self.dataset = dataset = 'intersection'
        semantic_models = ['word2vec-google-news']
        files_semantic_labels = []
        dim_out = 64
        str_aux = ''
        ds_var = None
        self.photo_dir = photo_dir = 'images'
        self.sketch_dir = sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        self.im_sz = im_sz = 224
        self.sk_sz = sk_sz = 224

        # if '_' in dataset: # 데이터 셋은 intersection으로 고정이기 때문에 이 if문 삭제
        #     token = dataset.split('_')
        #     dataset = token[0]
        #     ds_var = token[1]

        # semantic_models = sorted(semantic_models) # semantic_models 타입이 리스트인데 이 리스트에 하나밖에 없기 때문에 sort 필요 없음
        model_name = '+'.join(semantic_models)
        print('model_name : ' + model_name)
        # 데이터 셋 경로
        self.root_path = root_path = os.path.join(path_dataset, dataset)
        # 스케치 모델 경로 # 스케치에 대한 pth 경로
        path_sketch_model = os.path.join(path_aux, 'CheckPoints', dataset, 'sketch')
        print('path_sketch_model : ' + path_sketch_model)
        # 썸네일 모델 경로 # 이미지에 대한 pth 경로
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
        self.transform_image = transforms.Compose([transforms.Resize((im_sz, im_sz)), transforms.ToTensor()])
        self.transform_sketch = transforms.Compose([transforms.Resize((sk_sz, sk_sz)), transforms.ToTensor()])

        print('Loading data ...')
        splits = self._load_files_tuberlin_zeroshot(root_path=root_path, split_eccv_2018=False,
                                                                  photo_dir=photo_dir, sketch_dir=sketch_dir,
                                                                  photo_sd=photo_sd,
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

        print('te_fls_im type.{}'.format(type(splits['te_fls_im'])))
        self.splits_test = splits['te_fls_im']
        self.splits_train = splits['tr_fls_im']
        dict_clss = self._create_dict_texts(splits['tr_clss_im'])

        data_test_image = data_generator_image(self.dataset, root_path, photo_dir, photo_sd, splits['tr_fls_im'],
                                               splits['tr_clss_im'], transforms=self.transform_image)
        print('Done')

        # PyTorch test loader for sketch
        # test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=args.batch_size, shuffle=False,
        #                                 num_workers=args.num_workers, pin_memory=True)
        # PyTorch test loader for image
        self.test_loader_image = DataLoader(dataset=data_test_image, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers, pin_memory=True)

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
        print('tr_clss_len:', splits["tr_clss_im"].shape)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        sketch_embedding = ''
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
            preprocessed_data = Image.open(BytesIO(preprocessed_data))
            preprocessed_data = ImageOps.invert(preprocessed_data).convert(mode='RGB')
            transform_image = self.transform_sketch(preprocessed_data)
            print('transform_image shape : {}'.format(transform_image.shape))
            if torch.cuda.is_available():
                transform_image = transform_image.cuda()
                transform_image = transform_image.resize(1, 3, self.sk_sz, self.sk_sz)
                sketch_embedding = self.model.get_sketch_embeddings(transform_image)
            else:
                print('cuda is not available for image transforming')

            print('sketch_embedding shape : {}'.format(sketch_embedding.shape))

        else:
            print('input data error')
        return sketch_embedding

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        model_input = model_input.cpu().data.numpy()
        # acc_sk_em = np.concatenate(([], test_input_em), axis=0)
        print('test_input_em success shape : {}'.format(model_input.shape))

        # Compute mAP
        print('Computing evaluation metrics...', end='')

        # Compute similarity
        sim_euc = np.exp(-cdist(model_input, self.image_emd, metric='euclidean'))
        print('sim_euc shape : {}'.format(sim_euc.shape))
        print('Done')

        return sim_euc

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        dir_im = os.path.join(self.root_path, self.photo_dir, self.photo_sd)
        fls_im = np.asarray(self.splits_train)
        # print(fls_im)
        print(type(fls_im))
        print('fls_im size : {}'.format(len(fls_im)))

        postprocess_output = []

        ind_sk = np.argsort(inference_output)[0][:20]
        print('ind_sk shape {}'.format(ind_sk.shape))
        for j, iim in enumerate(ind_sk):
            print('iim : {}'.format(iim))
            filename = fls_im[iim].split("/")[-1]
            id = filename.split('.')[0]
            postprocess_output.append(fls_im[iim])
            # im = Image.open(os.path.join(dir_im, fls_im[iim])).convert(mode='RGB').resize(self.im_sz)
            # im.save(os.path.join(os.getcwd(), str(j + 1) + '.png'))
        print(postprocess_output)
        return [postprocess_output]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

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

        print('clss_im size : {}'.format(len(clss_im)))

        print('fls_sk size : {}'.format(len(fls_sk)))
        print('clss_sk size : {}'.format(len(clss_sk)))

        print('names_sk size : {}'.format(len(names_sk)))

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
            elif set_type == 'service':
                pass
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

        idx_tr_im, idx_tr_sk = self._get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='service')
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

