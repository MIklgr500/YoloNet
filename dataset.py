import os
import random
import imageio
import cv2
import copy

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

        self.root_dir       = config.root_dir
        self.img_dir        = os.path.join(config.root_dir, 'JPEGImages/')
        self.ann_dir        = os.path.join(config.root_dir, 'Annotations')
        self.cat_list       = config.cat_list
        self.class2label    = {l:i for i,l in enumerate(self.cat_list)}
        self.label2class    = {i:l for i,l in enumerate(self.cat_list)}
        self.train_size     = train_size
        self.ignore_list    = config.ignore_list
        self.img_size       = config.img_size
        self.NAnchors       = config.NAnchors
        self.ncell_in_grid  = config.ncell_in_grid
        self.scale_coeff    = config.scale_coeff
        self.flip_flag      = config.flip_flag
        self._base_cell     = { 'object':None,
                                'x_min':None,
                                'y_min':None,
                                'x_max':None,
                                'y_max':None
        }
        self.dataset        = { 'full':{},
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
            w = int(size.findChildren('width')[0].contents[0])
            h = int(size.findChildren('height')[0].contents[0])
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
        self.dataset['valid'] = crange

    def _kmeans(self, X):
        k = self.NAnchors
        print('kmeans with k = {0}'.format(k))
        x_num = X.shape[0]
        iterations = 0
        self.dataset['train'].append(ell)
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
        self.dataset            = dataset.dataset
        self.ncell_in_grid      = dataset.ncell_in_grid
        self.utype              = utype
        self.batch_size         = batch_size
        self.img_size           = dataset.img_size
        self.img_dir            = dataset.img_dir
        self.length             = len(dataset.dataset[str(utype)])
        self.random_state       = random_state
        self.flip_flag          = dataset.flip_flag
        self.scale_coeff        = dataset.scale_coeff
        self.grid_w             = self.img_size[0]//self.ncell_in_grid
        self.grid_h             = self.img_size[1]//self.ncell_in_grid
        self.nb_box             = dataset.NAnchors//2
        self.cat_list           = dataset.cat_list
        self.max_box_per_image  = dataset.max_box_per_image
        self.class2label        = dataset.class2label
        self.label2class        = dataset.label2class
        self.anchors            = [BoundBox(0,
                                            0,
                                            self.dataset.centroids[2*i],
                                            self.dataset.centroids[2*i+1])
                                   for i in range(self.nb_box))]
        super(VOC2007Generator, self).__init__(**kwargs)

    def _load_img(self, filename):
        img_path = os.path.join(self.img_dir, filename)
        img = imageio.imread(img_path).astype(np.float32)
        return cv2.resize(img,
                          (self.img_size[0], self.img_size[1]),
                          interpolation=cv2.INTER_CUBIC)

    def _get_bounds(self, i):
        lb, rb = int(i*self.batch_size), int((i+1)*self.batch_size)

        if rb > self.length:
            rb = self.length
            lb = rb - self.batch_size
        return lb, rb

    def __getitem__(self, index):
        x_batch = np.zeros((r_bound - l_bound, self.img_size[0], self.img_size[1], 3))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1,self.max_box_per_image, 4))
        y_batch = np.zeros((r_bound - l_bound, self.grid_h, self.grid_w, self.nb_box, 4+1+len(self.cat_list)))
        lb, rb = self._get_bounds(i)
        instance_count = 0
        for i in range(lb, rb):
            img, params = self._get_img(i)
            true_box_index = 0
            for box, grid_x, grid_y, best_anchor, obj_index in self._get_label(i, params):
                y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1

                b_batch[instance_count, 0, 0, 0, true_box_index] = box
                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image
            x_batch[instance_count] = img
            instance_count += 1
        return [x_batch, b_batch], y_batch

    def _get_img(self, i):
        fname = self.dataset[str(self.utype)][i]
        img = self._load_img(fname)

        random_params = {}
        random_params['scale'] = random.uniform()/self.scale_coeff+1
        img = cv2.resize(img, (0, 0), fx = random_params['scale'], fy = random_params['scale'])
        random_params['offx'] = int(random.uniform()*(random_params['scale']-1)*self.img_size[0])
        random_params['offy'] = int(random.uniform()*(random_params['scale']-1)*self.img_size[1])

        img = img[random_params['offx']:random_params['offx']+self.img_size[0],
                  random_params['offy']:random_params['offy']+self.img_size[1]],
                  :]

        if self.flip_flag:
            flip = random.uniform()
            if flip > 0.5:
                img = cv2.flip(img, 1)
                random_params['flip'] = True
        return img, random_params

    def _get_label(self, i, params):
        fname = self.dataset[str(self.utype)][i]

        all_objs = copy.deepcopy(train_instance['object'])

        scale = params['scale']
        offx = params['offx']
        offy = params['offy']
        flip = params['flip']

        for obj in all_objs:
            # augmentation
            for attr in ['xmin', 'xmax']:
                obj[attr] = int(obj[attr]*scale - offx)
                obj[attr] = max(min(obj[attr], 1), 0)

            for attr in ['ymin', 'ymax']:
                obj[attr] = int(obj[attr]*scale - offy)
                obj[attr] = max(min(obj[attr], 1), 0)

            if flip:
                obj['xmin'] += obj['xmax']
                obj['xmax'] = obj['xmin'] - obj['xmax']
                obj['xmin'] -= obj['xmax']

            center_x = 0.5*(obj['xmin']+obj['xmax'])/self.grid_w
            center_y = 0.5*(obj['xmin']+obj['xmax'])/self.grid_h

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            if grid_x < self.grid_h and grid_y < self.grid_w:
                obj_index = self.class2label[obj['object']]
                center_w = (obj['xmin']+obj['xmax'])/self.grid_w
                center_h = (obj['ymin']+obj['ymax'])/self.grid_h

                box = [center_x, center_y, center_w, center_h]

                best_anchor = -1
                max_iou     = -1

                shifted_box = BoundBox(0,
                                       0,
                                       center_w,
                                       center_h)
                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou    = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        best_anchor = i
                        max_iou = iou
                yield box, grid_x, grid_y, best_anchor, obj_index

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
