import numpy as np
import math
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.datasets import cifar10, cifar100


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



def pack_images(images,bs,col=None):


    images=tf.keras.backend.eval(images)
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)

    #images=images.transpose(0,3,1,2)

    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,H,W,C = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    pack = np.zeros( ( H*row, W*col,C), dtype=np.uint8 )
    for idx, img in enumerate(images):
        img = tf.keras.preprocessing.image.array_to_img(img)
        img = tf.keras.preprocessing.image.img_to_array(img)

        h = (idx//col) * H
        w = (idx% col) * W
        pack[h:h+H, w:w+W,:] = img

    pack_tensor=tf.convert_to_tensor(pack)
    return pack_tensor
    
def load_DI(traindataset_path,batch_size):


    train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        traindataset_path,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator