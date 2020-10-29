import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet



def load_model(model_path):

    if '.h5' in model_path:
        model= tf.keras.models.load_model(model_path)
    else:
        if 'resnet50v2' in model_path:
            model = ResNet50V2(weights='imagenet')
        elif 'mobilenet' in model_path:
            model = MobileNet(weights='imagenet')
        elif 'mobilenetv2' in model_path:
            model = MobileNetV2(weights='imagenet')
        elif 'vgg19' in model_path:
            model = VGG19(weights='imagenet')
        else:
            raise ValueError('Invalid model name : {}'.format(model_path))
    return model


def data_info(dataset):
    if dataset=='imagenet':
        num_classes=1000
        target_label=[1, 933, 946, 980, 107, 985, 151, 154, 207, 309, 311, 325, 340, 360, 386, 402, \
            403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963, 967, 574, 487]
        img_resol=224
        lower_res=2
    elif dataset =='cifar10':
        num_classes=10
        target_label= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        img_resol=32
        lower_res=1
    elif dataset== 'cifar100':
        num_classes=100
        target_label = []
        img_resol=32
        lower_res=1

    return num_classes, target_label,img_resol, lower_res

