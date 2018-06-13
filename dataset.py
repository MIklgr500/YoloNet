import os
import random
import imageio
import cv2

import numpy as np

from bs4 import BeautifulSoup
from tqdm import tqdm
from keras.utils import Sequence

from utils import IOU, avg_IOU

class VOC2007Dataset:
    """
    VOC2007 Dataset class for processing data
    and generate batch for train and validation step
        Arguments:
            config: Config, class where stored all parameters
    """
    def __init__(self, config=None, *args, **kwargs):
        assert config is not None, "Init. with None config"
        self.root_dir = config.root_dir
        self.img_dir = os.path.join(config.root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(config.root_dir, 'Annotations')
        self.cet_list = config.cat_list
        self.train_size = train_size
        self.ignore_list = config.ignore_list
        self.img_size = config.img_size
        self.NAnchors = config.NAnchors
        self.ncell_in_grid = config.ncell_in_grid
        self._base_cell = {
            'object':None,
            'x_min':None,
            'y_min':None,
            'x_max':None,
            'y_max':None
        }
        self.dataset = {
            'full':{},
            'train':[],
            'valid':[]
        }
        self._annotation_parser()
        self._train_validation_spliter()
        super(VOC2007Dataset, self).__init__(*args,**kwargs)

    def _xml2bs(self, filename):
        xml_path = os.path.join(self.ann_dir, filename)
        xml = ""
        with open(xml_path) as f:
            xml = f.readlines()
        xml = "".join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml)

    def _annotation_parser(self):
        for fn in tqdm(os.listdir(self.ann_dir), ascii=True, desc='Parse Annotation'):
            fname = fn[:-4]
            if fname in self.ignore_list:
                print('Skipping name:', name)
                continue
            ann = self._xml2bs(fn)
            objs = ann.findAll('object'):
            size = ann.findChildren('size')
            w = size.findChildren('width')[0].contents[0]
            h = size.findChildren('height')[0].contents[0]
            fname = str(ann.findChild('filename').contents[0])
            self.dataset['full'][fname] = []
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    cname = str(name_tag.contents[0])
                    if cname in self.cat_list:
                        cell = self._base_cell.copy()
                        bbox = obj.findChildren('bndbox')[0]
                        cell['object'] = str(cname)
                        cell['xmin'] = int(self.img_size[0]*bbox.findChildren('xmin')[0].contents[0]/w)
                        cell['ymin'] = int(self.img_size[1]*bbox.findChildren('ymin')[0].contents[0]/h)
                        cell['xmax'] = int(self.img_size[0]*bbox.findChildren('xmax')[0].contents[0]/w)
                        cell['ymax'] = int(self.img_size[1]*bbox.findChildren('ymax')[0].contents[0]/h)
                        self.dataset['full'][fname].append(cell)

    def _train_validation_spliter(self):
        clenght = len(self.dataset['full'])
        crange = [fname for fname in self.dataset['full']]
        while len(crange)>int((1-self.train_size)*clenght):
            i = random.choice(list(range(len(crange))))
            ell = crange.pop(i)
            self.dataset['train'].append(ell)
        self.dataset['valid'] = crange

    def _kmeans(self, X):
        k = self.NAnchors
        print('kmeans with k = {0}'.format(k))
        x_num = X.shape[0]
        iterations = 0
        prev_assignments = np.ones(k)*(-1)
        iteration = 0
        old_distances = np.zeros((x_num, k))

        indices = [random.randrange(x_num) for i in range(k)]
        centroids = X[indices]
        anchor_dim = X.shape[1]

        while True:
            distances = []
            iteration += 1
            for i in range(x_num):
                d = 1 - IOU(ann_dims[i], centroids)
                distances.append(d)
            distances = np.array(distances)
            print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

            #assign samples to centroids
            assignments = np.argmin(distances,axis=1)

            if (assignments == prev_assignments).all() :
                return centroids

            #calculate new centroids
            centroid_sums=np.zeros((k, anchor_dim), np.float)
            for i in range(x_num):
                centroid_sums[assignments[i]]+=X[i]
            for j in range(k):
                centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

            prev_assignments = assignments.copy()
            old_distances = distances.copy()

    def get_anchors(self):
        X = []
        for fname in self.dataset['full']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/self.ncell_in_grid
            relative_h = (float(obj['ymax']) - float(obj['ymin']))/self.ncell_in_grid
            X.append(tuple(map(float, (relative_w, relative_h))))

        X = np.array(X)
        self.centroids = self._kmeans(X)

    @property
    def centroids(self):
        return self.centroids

class VOC2007Generator(Sequence):
    """
    VOC2007Generator class: batch Generator
        Arguments:
            dataset:VOC2007Datatset
            type: train or valid
            batch_size: int - number samples in one batch
    """
    def __init__(self, dataset, batch_size=8, utype='train', random_state=134, **kwargs):
        self.dataset = dataset.dataset
        self.utype = utype
        self.batch_size = batch_size
        self.img_size = dataset.img_size
        self.img_dir = dataset.img_dir
        self.length = len(dataset.dataset[str(utype)])
        self.random_state = random_state
        super(VOC2007Generator, self).__init__(**kwargs)

    def _load_img(self, filename):
        img_path = os.path.join(self.img_dir, filename)
        img = imageio.imread(img_path).astype(np.float32)
        return cv2.resize(img,
                          (self.img_size[0], self.img_size[1]),
                          interpolation=cv2.INTER_CUBIC)

    def __getitem__(self, index):
        img_batch = []
        label_batch = []
        if self.utype=='train':
            for i in range(int(self.batch_size*index), int(self.batch_size*(index+1))):
                random_params = self._get_random_params()
                img_batch.append(self._get_img(i, random_params))
                label_batch.append(self._get_label(i, random_params))
            return img_batch, label_batch
        elif self.utype=='valid':
            for i in range(int(self.batch_size*index), int(self.batch_size*(index+1))):
                img_batch.append(self._get_img(i, random_params))
            return img_batch
               
    def _get_img(self, i, random_params):
        pass

    def _get_label(self, i, random_params):
        pass

    def _get_random_params(self):
        pass

    def __len__(self):
        return int(np.ceil(float(self.length)/self.batch_size))

    def on_epoch_end(self):
        """
            Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """
            Create an infinite generator that iterate over the Sequence.
        """
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
            self.on_epoch_end()
