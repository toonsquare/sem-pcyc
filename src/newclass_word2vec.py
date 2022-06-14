import glob
import os
import re

import gensim
import numpy as np

import utils

# root_path = "/home/ubuntu/sem_pcyc/dataset/intersection/images"
# root_path = "/home/model-server/sem_pcyc/dataset/intersection/images"
root_path = "/home/ubuntu/sem_pcyc/data/sem_pcyc/dataset/intersection/images"

def create_wordemb(root_path, type='word2vec-google-news-300'):
    """
    이미지의 클래스가 추가되었을 경우, word to vector를 통해 새로운 vector 값을 만들어주는 함수
    """
    clss = glob.glob(os.path.join(root_path, '*'))
    clss = [c.split('/')[-1] for c in clss if os.path.isdir(c)]
    available_type = ['word2vec-google-news-300', 'fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300']
    if type not in available_type:
        print("Type specified does not exist.")
        exit()
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     '/home/model-server/GoogleNews-vectors-negative300.bin', binary=True).wv

    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/ubuntu/sem_pcyc/GoogleNews-vectors-negative300.bin', binary=True).wv

    lv = len(model['airplane'])  # for knowing the length of vector

    wordemb = dict()
    # synonym dictionary
    syn_dict = utils.get_synonym()
    for cls in clss:
        cls_tmp = re.sub('[)(]', '', cls)
        if cls_tmp in model.vocab:
            ws = [cls_tmp]
        elif cls_tmp in syn_dict.keys():
            ws = syn_dict[cls_tmp].split('_')
        else:
            ws = cls_tmp.split('_')
        v = np.zeros((len(ws), lv))
        for i, w in enumerate(ws):
            if w in model.vocab:
                v[i, :] = model[w]
            else:
                print(w)
        wordemb.update({cls: np.mean(v, axis=0)})
    return wordemb


plus_word = {}

plus_word = create_wordemb(root_path)

# 생성한 dict를 npy로 만들기
# np.save('/home/model-server/sem_pcyc/aux/Semantic/intersection/new_plus_words.npy', plus_word)
print('making npy.... ')
np.save('/home/ubuntu/sem_pcyc/data/sem_pcyc/aux/Semantic/intersection/new_plus_words.npy', plus_word)
print('npy create!')

'''
# 생성한 npy 불러오기 - 클래스 확인용
load_dict = np.load('/home/model-server/sem_pcyc/aux/Semantic/intersection/new_plus_words.npy', allow_pickle=True)

load_dict_list = load_dict.tolist()
dict_list = []
for key, value in load_dict_list.items():
    dict_list.append(key)
print(dict_list)
print(len(dict_list))
print(type(load_dict_list))
'''
