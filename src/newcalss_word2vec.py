import glob
import os
import re

import gensim
import numpy as np

import utils

# all_word_set = ['alley', 'backyard', 'balloon', 'bedroom', 'bike_racks', 'bistro', 'board', 'bottle', 'building', 'bus', 'cabinet', 'cafe', 'car',
#                  'card', 'cell', 'chalkboard', 'charger', 'classroom', 'cone', 'detergent', 'division', 'doll', 'faucet', 'fishing_rod',
#                 'forest', 'gear', 'gym', 'handle', 'hanger', 'jesus', 'kettle', 'kitchen', 'kneel', 'laundry', 'leash', 'library', 'lie', 'livingroom',
#                 'forest', 'gear', 'gym', 'handle', 'hanger', 'jesus', 'kettle', 'kitchen', 'kneel', 'laundry', 'leash', 'library', 'lie', 'livingroom',
#                 'market', 'minus', 'multiplication', 'office', 'padlock', 'park', 'passport', 'pavilion', 'platform', 'plunger', 'plus',
#                 'remote', 'root', 'school', 'sofa', 'sign', 'standing', 'station', 'studio', 'study', 'sunglass', 'television', 'ticket', 'toothpaste', 'tower', 'trap',
#                 'treelined', 'underwater', 'window', 'worktable']
# all word set 개수: 68개


root_path = "/home/ubuntu/sem_pcyc/dataset/intersection/images"


def create_wordemb(root_path, type='word2vec-google-news-300'):
    clss = glob.glob(os.path.join(root_path, '*'))
    clss = [c.split('/')[-1] for c in clss if os.path.isdir(c)]
    available_type = ['word2vec-google-news-300', 'fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300']
    if type not in available_type:
        print("Type specified does not exist.")
        exit()
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/ubuntu/projects_paul/GoogleNews-vectors-negative300.bin', binary=True).wv

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
# for i in range(len(all_word_set)):
#     print(all_word_set[i])
#     if all_word_set[i] == 'baseball_bat' :
#         pass
#     else:
#         embedding = model[all_word_set[i]]
#         print(i)
#         plus_word[i] = embedding
#     # print(plus_word)
#     # print(f'{i} : {embedding}')
#     # print('#' * 50)
# # print(plus_word)

plus_word = create_wordemb(root_path)

# 생성한 dict를 npy로 만들기
np.save('intersection_plus_words.npy', plus_word)
print('npy create!')

# 생성한 npy 불러오기
load_dict = np.load('plus_words.npy', allow_pickle=True)
print(load_dict)
print(type(load_dict))

print(load_dict.shape)

load_dict_list = load_dict.tolist()
dict_list = []
for key, value in load_dict_list.items():
    dict_list.append(key)
print(dict_list)
print(len(dict_list))
print(type(load_dict_list))

# 2개의 dictionary merge하기
# new_word2vec = dict(plus_words, **word2vec)
