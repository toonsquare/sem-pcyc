import numpy as np
import utils
import os

config = utils.read_config()
path_dataset = config['path_dataset']
path_aux = config['path_aux']
images = os.path.join(path_dataset, 'intersection', 'images')
sketches = os.path.join(path_dataset, 'intersection', 'sketches')

clss = []
npys = [ '/ml_data/sem_pcyc/aux/Semantic/intersection/plus_words.npy', '/ml_data/sem_pcyc/aux/Semantic/intersection/word2vec-google-news.npy']


for npy in npys:
    print(npy)
    load_dict = np.load(npy, allow_pickle=True)
    for key, value in load_dict.tolist().items():
        clss.append(key)
        if key == 'plunger':
            print('----------------plunger---------------------')
            print(value)
print(len(clss))
print(clss)
print(path_dataset)
image_list =  os.listdir(images)
print('image list : {}'.format(image_list))
print ('-------image diff --------------')
diff = set(image_list) - set(clss)
print(diff)

sketches_list =  os.listdir(sketches)
print('sketches list : {}'.format(sketches_list))
print ('-------sketches diff --------------')
diff = set(sketches_list) - set(clss)
print(diff)