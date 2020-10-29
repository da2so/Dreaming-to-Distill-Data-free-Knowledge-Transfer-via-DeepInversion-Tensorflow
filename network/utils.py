import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

from network.resnet import resnet_34 ,resnet_18 ,resnet_50


def load_dataset(dataset_name, batch_size):

    if dataset_name =='cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes=10

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes=100
    else:
        raise ValueError('Invalid dataset name : {}'.format(dataset_name))
    

    # subtract mean and normalize
    mean_image = np.mean(x_train, axis=0)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    y_train  = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return x_train ,y_train , x_test, y_test,num_classes


def load_teacher(model_name,img_shape,num_classes):

    if model_name == 'resnet34':
        teacher_model = resnet_34(img_shape,num_classes)
    elif model_name == 'resnet18':
        teacher_model = resnet_18(img_shape, num_classes)
    elif model_name == 'resnet50':
        teacher_model = resnet_50(img_shape, num_classes)

    else:
        raise ValueError('Invalid model name : {}'.format(model_name))

    return teacher_model