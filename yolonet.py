import os
import cv2

import numpy as np

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPool2D, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import tf
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from dataset import VOC2007Generator
from utils import decode_netout, compute_overlap, compute_ap
from basenet import MobileNetFeature, MobileNetV2Feature

class YOLONet(object):
    """
        YoLO neural network

    """
    def __init__(self,
                 config,
                 anchor
                 backend='MobileNet'):
        self.input_size = config.img_size[0]
        self.nb_class = len(config.cat_list)
        self.nb_box = config.NAnchors//2
        self.labels = list(comfig.cat_list)
        self.class_w = np.ones(self.nb_class, dtype='float32')
        self.anchor = anchor

        self.max_box_per_image = config.max_box_per_image

        input_img = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxs = Input(shape=(1,1,1,1, max_box_per_image, 4))

        if backend=='MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif backend=='MobileNetV2':
            self.feature_extractor = MobileNetV2Feature(self.input_size)
        else:
            raise Exception('Please, check your BaseNet')

        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.feature_extract(input_img)

        out = Conv2D(self.nb_box*(4+1+self.nb_class)),
                     (1,1), strides=(1,1),
                     padding='same'
                     name='DetectionLayer',
                     kernel_initializer='lecun_normal')(features)
        out = Reshape((sefl.grid_h, self.grid_w, self.nb_box, 4+1+self.nb_class))(out)
        out = Lambda(lambda args: args[0])([out, self.true_boxs])

        self.model = Model([input_img, self.true_boxs], out)

        layer = sefl.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0]/(self.grid_h*self.grid_w))
        new_bias = np.random.normal(size=weights[1]/(self.grid_h*self.grid_w))

        layer.set_weights([new_kernel, new_bias])

        self.model.summary()

    def _custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        call_y = tf.transpose(cell_X, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        pred_box_xy = tf.sigmoid(y_pred[..., :2])+cell_grid
        pred_box_wh = tf.exp(y_pred[..., 2:4])*np.reshape(self.anchors, [1,1,1,self.nb_box, 2])
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_class = y_pred[..., 5:]

        true_box_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]

        true_wh_half = true_box_wh/2
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh/2
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes-intersect_mins, 0.)

        intersect_areas = intersect_wh[..., 0]*intersect_wh[..., 1]
        true_areas = true_box_wh[..., 0]*true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0]*pred_box_wh[..., 1]

        union_areas = pred_areas+true_areas-intersect_areas
        iou_score = tf.true_div(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_score, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6)*(1-y_true[..., 4])*self.no_object_scale
        conf_mask = conf_mask + y_true[..., 4]*self.object_scale

        class_mask = y_true[..., 4]*tf.gather(self.class_wt, true_box_class)*self.class_scale

        no_box_mask = tf.to_float(coord_mask < self.coord_scale/2.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1),
                                                       lambda: [true_box_xy+(0.5+cell_grid)*no_box_mask,
                                                                true_box_wh+tf.ones_like(true_box_wh)*\
                                                                np.reshape(self.anchors, [1,1,1,self.nb_box,2])*\
                                                                no_boxes_mask, tf.ones_like(coord_mask)],
                                                       lambda: [
                                                            true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)* coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)* coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches+1),
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def train(self, train_gen,     # the list of images to train the model
                    valid_gen,     # the list of images used to validate the model
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    steps_per_epoch,
                    valid_steps_per_epoch,
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    saved_weights_name='best_weights.h5',
                    debug=False):
        self.batch_size = batch_size

        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        self.debug = debug

        train_generator = train_gen
        valid_generator = valid_gen

        self.warmup_batches  = warmup_epochs * (steps_per_epoch + valid_steps_per_epoch)

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  #write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        self.model.fit_generator(generator = train_generator,
                                 steps_per_epoch = steps_per_epoch,
                                 epochs = warmup_epochs + nb_epochs,
                                 verbose = 2 if debug else 1,
                                 validation_data = valid_generator,
                                 validation_steps = valid_steps_per_epoch,
                                 callbacks = [early_stop, checkpoint, tensorboard],
                                 workers = 3,
                                 max_queue_size = 8)

        average_precisions = self.evaluate(valid_generator)

        # print evaluation
        for label, average_precision in average_precisions.items():
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes
