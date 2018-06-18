from keras.models import Model
from keras.layers import Input
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2


class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        mobilenet = MobileNet(input_shape=(224,224,3),
                              include_top=False,
                              weights='imagenet')

        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image

class MobileNetV2Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        mobilenet = MobileNetV2(input_shape=(224,224,3),
                                include_top=False,
                                weights='imagenet')

        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image
