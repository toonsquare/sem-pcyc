import numpy as np
from models import DataGeneratorImage, SEM_PCYC
import torchvision.transforms as transforms
import os
import glob
import random
import itertools
import torch
from torch.utils.data import DataLoader

class MakeNPY():
    def __init__(self):
        self.path_dataset = '/home/ubuntu/sem_pcyc/dataset'
        self.path_aux = '/home/ubuntu/sem_pcyc/aux'
        self.dataset = 'intersection'
        self.root_path  = os.path.join(self.path_dataset, self.dataset)
        self.photo_dir = 'images'
        self.sketch_dir = 'sketches'
        self.photo_sd = ''
        self.sketch_sd = ''
        self.splits = self._load_files_tuberlin_zeroshot(root_path=self.root_path, split_eccv_2018=False,
                                                       photo_dir=self.photo_dir, sketch_dir=self.sketch_dir, photo_sd=self.photo_sd,
                                                       sketch_sd=self.sketch_sd,dataset=self.dataset)
        self.sem_pcyc_model = self.SEM_PCYC()
    def _get_coarse_grained_samples(self,classes, fls_im, fls_sk, set_type='train', filter_sketch=True):
        idx_im_ret = np.array([], dtype=np.int)
        idx_sk_ret = np.array([], dtype=np.int)
        clss_im = np.array([f.split('/')[-2] for f in fls_im])
        clss_sk = np.array([f.split('/')[-2] for f in fls_sk])
        names_sk = np.array([f.split('-')[0] for f in fls_sk])
        for i, c in enumerate(classes):
            idx1 = np.where(clss_im == c)[0]
            idx2 = np.where(clss_sk == c)[0]
            if set_type == 'train':
                idx_cp = list(itertools.product(idx1, idx2))
                if len(idx_cp) > 100000:
                    random.seed(i)
                    idx_cp = random.sample(idx_cp, 100000)
                idx1, idx2 = zip(*idx_cp)
            # elif set_type == 'service':
            #    pass
            else:
                # remove duplicate sketches
                if filter_sketch:
                    names_sk_tmp = names_sk[idx2]
                    idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                    idx2 = idx2[idx_tmp]
            idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
            idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)

        return idx_im_ret, idx_sk_ret

    def _load_files_tuberlin_zeroshot( self,root_path, split_eccv_2018=False, photo_dir='images', sketch_dir='sketches',
                                      photo_sd='', sketch_sd='', dataset=''):
        path_im = os.path.join(root_path, photo_dir, photo_sd)
        path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

        # image files and classes
        if dataset == '':
            fls_im = glob.glob(os.path.join(path_im, '*', '*'))
        else:
            fls_im = glob.glob(os.path.join(path_im, '*', '*.base64'))

        fls_im = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im])
        clss_im = np.array([f.split('/')[-2] for f in fls_im])

        # sketch files and classes
        fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
        fls_sk = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk])
        clss_sk = np.array([f.split('/')[-2] for f in fls_sk])

        # all the unique classes
        classes = np.unique(clss_im)

        # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
        np.random.seed(0)
        tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False)
        va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.06 * len(classes)), replace=False)
        # te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))
        te_classes = np.random.choice(classes, int(1 * len(classes)), replace=False)

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

    def images_preprocessing(self):
        image_test = self.splits


        # 클래스명이 필요하기 때문에 image_test라는 튜플의 첫 번째 인덱스를 all_clss_im으로 저장
        # str_sim이 필요가 없지만 DataGeneratorImage 클래스에서는 clss_im이 사용되므로 일단 구해놓기
        all_class_image = image_test["te_clss_im"]
        all_files_image = image_test["te_fls_im"]

        print(all_files_image)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        # 이미지들을 tensor화 하는 단계
        data_test_image = DataGeneratorImage(self.dataset, self.root_path, self.photo_dir, self.photo_sd, all_files_image,
                                             all_class_image, transforms=transform)

        # tensor화된 이미지들을 load하는 단계
        test_loader_image = DataLoader(dataset=data_test_image, batch_size=32, shuffle=False, num_workers=4,
                                       pin_memory=True)

        # 데이터를 enumerate()를 통해 im(이미지)와 cls_im 변수에 각각 vertor와 클래스명들을 담아준다.
        print("start embedding")
        for i, (im, cls_im) in enumerate(test_loader_image):
            if torch.cuda.is_available():
                im = im.cuda()
            # print("Present image number :", i)

            im_em = self.sem_pcyc_model.get_image_embeddings(im)

            if i == 0:
                acc_im_em = im_em.cpu().data.numpy()
                acc_cls_im = cls_im
            else:  # acc_im_em과 im_em을 concatenate한 것인데, 이것을 cpu로 담고 numpy() 배열로 바꿔주어 추후 cdist에 사용할 수 있게된다.
                acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)  # 유클리디안
                acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)  # str_sim에 사용 -> 필요 없음

        return acc_im_em

    def _create_dict_texts(self,texts):
        texts = sorted(list(set(texts)))
        d = {l: i for i, l in enumerate(texts)}
        return d
    def SEM_PCYC(self):
        path_sketch_model = os.path.join(self.path_aux, 'CheckPoints', self.dataset, 'sketch')
        path_image_model = os.path.join(self.path_aux, 'CheckPoints', self.dataset, 'image')
        dict_clss = self._create_dict_texts(self.splits['tr_clss_im'])

        files_semantic_labels = []

        sem_dim = 0
        semantic_models = ['new_plus_words']
        semantic_models = sorted(semantic_models)

        for f in semantic_models:
            fi = os.path.join(self.path_aux, 'Semantic', self.dataset, f + '.npy')
            print('fi : ' + fi)
            files_semantic_labels.append(fi)
            sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

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


        sem_pcyc_model = SEM_PCYC(params_model)
        path_pth ="/home/ubuntu/sem_pcyc/aux/CheckPoints/intersection/new_plus_words/64/model_best.pth"
        device = torch.device("cuda")
        checkpoint = torch.load(path_pth,map_location="cuda:0")
        sem_pcyc_model.load_state_dict(checkpoint['state_dict'])
        sem_pcyc_model.to(device)
        sem_pcyc_model.eval()
        return sem_pcyc_model

def main() :
    make_npy = MakeNPY()
    acc_im_em = make_npy.images_preprocessing()
    print("--------------END Embedding--------------")
    print('size acc_im_em : {}'.format(len(acc_im_em)))
    print("\n")
    print("--------------START Saving--------------")
    np.save("/home/ubuntu/projects_jonathan/acc_im_em.npy", acc_im_em)
    print("--------------END Saving--------------")

if __name__ == "__main__":
    main()