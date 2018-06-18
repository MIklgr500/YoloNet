class Config(object):
    """
    Config class
        Arguments:
            root_dir
            cat_list:some categorys from list=[
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train','tvmonitor'
            ] for VOC2007 dataset
            ignore_list: contain filename, which will be ignore
            train_size: (0, 1]-part data on train set
    """
    def __init__(self,
                 root_dir = 'VOCdevkit/VOCdevkit/VOC2012/',
                 cat_list=['person'],
                 ignore_list = [],
                 train_size = 0.75,
                 img_size=[224, 224, 3],
                 batch_size = 8,
                 ncell_in_grid = 16,
                 NAnchors = 10,
                 max_box_per_image = 10
                 ):
        # path and cat
        self.root_dir = root_dir
        self.ignore_list = ignore_list
        # network params
        self.cat_list = cat_list
        self.train_size = train_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.ncell_in_grid = ncell_in_grid
        self.NAnchors = NAnchors
        self.max_box_per_image = max_box_per_image
        # augmentation
        self.scale_coeff = 10.
        self.flip_flag = True
