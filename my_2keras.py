#!/usr/bin/env python3
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
# fbnet_pretrained_model = fbnet("fbnet_a", pretrained=True)


import tensorflow as tf
from tensorflow import keras
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable

def KerasNet(input_shape=(224, 224, 3)):
    image_input = keras.layers.Input(shape=input_shape)

    xif0_0 = keras.layers.Conv2D(
        16, (3, 3), strides=(2, 2), padding="same")(image_input)
    xif0_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif0_0)
    xif0_0 = keras.layers.Activation("relu")(xif0_0)

    xif1_0 = xif0_0

    #xif2_0
    # pw
    xif2_0 = keras.layers.Conv2D(
        48, (1, 1), strides=(1, 1))(xif1_0)
    xif2_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif2_0)
    xif2_0 = keras.layers.Activation("relu")(xif2_0)
    # dw
    xif2_0 = keras.layers.Conv2D(
        48, (3, 3), strides=(2, 2), padding="same", groups=48)(xif2_0)
    xif2_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif2_0)
    xif2_0 = keras.layers.Activation("relu")(xif2_0)
    # pwl
    xif2_0 = keras.layers.Conv2D(
        24, (1, 1), strides=(1, 1))(xif2_0)
    xif2_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif2_0)

    #xif2_1
    # dw
    xif2_1 = keras.layers.Conv2D(
        24, (3, 3), strides=(1, 1), padding="same", groups=24)(xif2_0)
    xif2_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif2_1)
    xif2_1 = keras.layers.Activation("relu")(xif2_1)
    # pwl
    xif2_1 = keras.layers.Conv2D(
        24, (1, 1), strides=(1, 1))(xif2_1)
    xif2_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif2_1)
    # res_conn
    xif2_1 = keras.layers.Add()([xif2_1,xif2_0])

    xif2_2 = xif2_1

    xif2_3 = xif2_2

    #xif3_0
    # pw
    xif3_0 = keras.layers.Conv2D(
        144, (1, 1), strides=(1, 1))(xif2_3)
    xif3_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_0)
    xif3_0 = keras.layers.Activation("relu")(xif3_0)
    # dw
    xif3_0 = keras.layers.Conv2D(
        144, (5, 5), strides=(2, 2), padding="same", groups=144)(xif3_0)
    xif3_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_0)
    xif3_0 = keras.layers.Activation("relu")(xif3_0)
    # pwl
    xif3_0 = keras.layers.Conv2D(
        32, (1, 1), strides=(1, 1))(xif3_0)
    xif3_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_0)

    #xif3_1
    # pw
    xif3_1 = keras.layers.Conv2D(
        96, (1, 1), strides=(1, 1))(xif3_0)
    xif3_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_1)
    xif3_1 = keras.layers.Activation("relu")(xif3_1)
    # dw
    xif3_1 = keras.layers.Conv2D(
        96, (3, 3), strides=(1, 1), padding="same", groups=96)(xif3_1)
    xif3_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_1)
    xif3_1 = keras.layers.Activation("relu")(xif3_1)
    # pwl
    xif3_1 = keras.layers.Conv2D(
        32, (1, 1), strides=(1, 1))(xif3_1)
    xif3_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_1)
    # res_conn
    xif3_1 = keras.layers.Add()([xif3_1,xif3_0])

    #xif3_2
    # dw
    xif3_2 = keras.layers.Conv2D(
        32, (5, 5), strides=(1, 1), padding="same", groups=32)(xif3_1)
    xif3_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_2)
    xif3_2 = keras.layers.Activation("relu")(xif3_2)
    # pwl
    xif3_2 = keras.layers.Conv2D(
        32, (1, 1), strides=(1, 1))(xif3_2)
    xif3_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_2)
    # res_conn
    xif3_2 = keras.layers.Add()([xif3_2,xif3_1])

    #xif3_3
    # pw
    xif3_3 = keras.layers.Conv2D(
        96, (1, 1), strides=(1, 1))(xif3_2)
    xif3_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_3)
    xif3_3 = keras.layers.Activation("relu")(xif3_3)
    # dw
    xif3_3 = keras.layers.Conv2D(
        96, (3, 3), strides=(1, 1), padding="same", groups=96)(xif3_3)
    xif3_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_3)
    xif3_3 = keras.layers.Activation("relu")(xif3_3)
    # pwl
    xif3_3 = keras.layers.Conv2D(
        32, (1, 1), strides=(1, 1))(xif3_3)
    xif3_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif3_3)
    # res_conn
    xif3_3 = keras.layers.Add()([xif3_3,xif3_2])

    #xif4_0
    # pw
    xif4_0 = keras.layers.Conv2D(
        192, (1, 1), strides=(1, 1))(xif3_3)
    xif4_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_0)
    xif4_0 = keras.layers.Activation("relu")(xif4_0)
    # dw
    xif4_0 = keras.layers.Conv2D(
        192, (5, 5), strides=(2, 2), padding="same", groups=192)(xif4_0)
    xif4_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_0)
    xif4_0 = keras.layers.Activation("relu")(xif4_0)
    # pwl
    xif4_0 = keras.layers.Conv2D(
        64, (1, 1), strides=(1, 1))(xif4_0)
    xif4_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_0)

    #xif4_1
    # pw
    xif4_1 = keras.layers.Conv2D(
        192, (1, 1), strides=(1, 1))(xif4_0)
    xif4_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_1)
    xif4_1 = keras.layers.Activation("relu")(xif4_1)
    # dw
    xif4_1 = keras.layers.Conv2D(
        192, (5, 5), strides=(1, 1), padding="same", groups=192)(xif4_1)
    xif4_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_1)
    xif4_1 = keras.layers.Activation("relu")(xif4_1)
    # pwl
    xif4_1 = keras.layers.Conv2D(
        64, (1, 1), strides=(1, 1))(xif4_1)
    xif4_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_1)
    # res_conn
    xif4_1 = keras.layers.Add()([xif4_1,xif4_0])

    #xif4_2
    # dw
    xif4_2 = keras.layers.Conv2D(
        64, (5, 5), strides=(1, 1), padding="same", groups=64)(xif4_1)
    xif4_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_2)
    xif4_2 = keras.layers.Activation("relu")(xif4_2)
    # pwl
    xif4_2 = keras.layers.Conv2D(
        64, (1, 1), strides=(1, 1), groups=2)(xif4_2)
    xif4_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_2)
    # res_conn
    xif4_2 = keras.layers.Add()([xif4_2,xif4_1])

    #xif4_3
    # pw
    xif4_3 = keras.layers.Conv2D(
        384, (1, 1), strides=(1, 1))(xif4_2)
    xif4_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_3)
    xif4_3 = keras.layers.Activation("relu")(xif4_3)
    # dw
    xif4_3 = keras.layers.Conv2D(
        384, (5, 5), strides=(1, 1), padding="same", groups=384)(xif4_3)
    xif4_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_3)
    xif4_3 = keras.layers.Activation("relu")(xif4_3)
    # pwl
    xif4_3 = keras.layers.Conv2D(
        64, (1, 1), strides=(1, 1))(xif4_3)
    xif4_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_3)
    # res_conn
    xif4_3 = keras.layers.Add()([xif4_3,xif4_2])

    #xif4_4
    # pw
    xif4_4 = keras.layers.Conv2D(
        384, (1, 1), strides=(1, 1))(xif4_3)
    xif4_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_4)
    xif4_4 = keras.layers.Activation("relu")(xif4_4)
    # dw
    xif4_4 = keras.layers.Conv2D(
        384, (3, 3), strides=(1, 1), padding="same", groups=384)(xif4_4)
    xif4_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_4)
    xif4_4 = keras.layers.Activation("relu")(xif4_4)
    # pwl
    xif4_4 = keras.layers.Conv2D(
        112, (1, 1), strides=(1, 1))(xif4_4)
    xif4_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_4)

    #xif4_5
    # dw
    xif4_5 = keras.layers.Conv2D(
        112, (5, 5), strides=(1, 1), padding="same", groups=112)(xif4_4)
    xif4_5 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_5)
    xif4_5 = keras.layers.Activation("relu")(xif4_5)
    # pwl
    xif4_5 = keras.layers.Conv2D(
        112, (1, 1), strides=(1, 1), groups=2)(xif4_5)
    xif4_5 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_5)
    # res_conn
    xif4_5 = keras.layers.Add()([xif4_5,xif4_4])

    #xif4_6
    # pw
    xif4_6 = keras.layers.Conv2D(
        336, (1, 1), strides=(1, 1))(xif4_5)
    xif4_6 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_6)
    xif4_6 = keras.layers.Activation("relu")(xif4_6)
    # dw
    xif4_6 = keras.layers.Conv2D(
        336, (5, 5), strides=(1, 1), padding="same", groups=336)(xif4_6)
    xif4_6 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_6)
    xif4_6 = keras.layers.Activation("relu")(xif4_6)
    # pwl
    xif4_6 = keras.layers.Conv2D(
        112, (1, 1), strides=(1, 1))(xif4_6)
    xif4_6 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_6)
    # res_conn
    xif4_6 = keras.layers.Add()([xif4_6,xif4_5])

    #xif4_7
    # dw
    xif4_7 = keras.layers.Conv2D(
        112, (3, 3), strides=(1, 1), padding="same", groups=112)(xif4_6)
    xif4_7 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_7)
    xif4_7 = keras.layers.Activation("relu")(xif4_7)
    # pwl
    xif4_7 = keras.layers.Conv2D(
        112, (1, 1), strides=(1, 1), groups=2)(xif4_7)
    xif4_7 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif4_7)
    # res_conn
    xif4_7 = keras.layers.Add()([xif4_7,xif4_6])

    #xif5_0
    # pw
    xif5_0 = keras.layers.Conv2D(
        672, (1, 1), strides=(1, 1))(xif4_7)
    xif5_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_0)
    xif5_0 = keras.layers.Activation("relu")(xif5_0)
    # dw
    xif5_0 = keras.layers.Conv2D(
        672, (5, 5), strides=(2, 2), padding="same", groups=672)(xif5_0)
    xif5_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_0)
    xif5_0 = keras.layers.Activation("relu")(xif5_0)
    # pwl
    xif5_0 = keras.layers.Conv2D(
        184, (1, 1), strides=(1, 1))(xif5_0)
    xif5_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_0)

    #xif5_1
    # pw
    xif5_1 = keras.layers.Conv2D(
        1104, (1, 1), strides=(1, 1))(xif5_0)
    xif5_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_1)
    xif5_1 = keras.layers.Activation("relu")(xif5_1)
    # dw
    xif5_1 = keras.layers.Conv2D(
        1104, (5, 5), strides=(1, 1), padding="same", groups=1104)(xif5_1)
    xif5_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_1)
    xif5_1 = keras.layers.Activation("relu")(xif5_1)
    # pwl
    xif5_1 = keras.layers.Conv2D(
        184, (1, 1), strides=(1, 1))(xif5_1)
    xif5_1 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_1)
    # res_conn
    xif5_1 = keras.layers.Add()([xif5_1,xif5_0])

    #xif5_2
    # pw
    xif5_2 = keras.layers.Conv2D(
        552, (1, 1), strides=(1, 1))(xif5_1)
    xif5_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_2)
    xif5_2 = keras.layers.Activation("relu")(xif5_2)
    # dw
    xif5_2 = keras.layers.Conv2D(
        552, (5, 5), strides=(1, 1), padding="same", groups=552)(xif5_2)
    xif5_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_2)
    xif5_2 = keras.layers.Activation("relu")(xif5_2)
    # pwl
    xif5_2 = keras.layers.Conv2D(
        184, (1, 1), strides=(1, 1))(xif5_2)
    xif5_2 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_2)
    # res_conn
    xif5_2 = keras.layers.Add()([xif5_2,xif5_1])

    #xif5_3
    # pw
    xif5_3 = keras.layers.Conv2D(
        1104, (1, 1), strides=(1, 1))(xif5_2)
    xif5_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_3)
    xif5_3 = keras.layers.Activation("relu")(xif5_3)
    # dw
    xif5_3 = keras.layers.Conv2D(
        1104, (5, 5), strides=(1, 1), padding="same", groups=1104)(xif5_3)
    xif5_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_3)
    xif5_3 = keras.layers.Activation("relu")(xif5_3)
    # pwl
    xif5_3 = keras.layers.Conv2D(
        184, (1, 1), strides=(1, 1))(xif5_3)
    xif5_3 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_3)
    # res_conn
    xif5_3 = keras.layers.Add()([xif5_3,xif5_2])

    #xif5_4
    # pw
    xif5_4 = keras.layers.Conv2D(
        1104, (1, 1), strides=(1, 1))(xif5_3)
    xif5_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_4)
    xif5_4 = keras.layers.Activation("relu")(xif5_4)
    # dw
    xif5_4 = keras.layers.Conv2D(
        1104, (5, 5), strides=(1, 1), padding="same", groups=1104)(xif5_4)
    xif5_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_4)
    xif5_4 = keras.layers.Activation("relu")(xif5_4)
    # pwl
    xif5_4 = keras.layers.Conv2D(
        352, (1, 1), strides=(1, 1))(xif5_4)
    xif5_4 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif5_4)

    #xif6_0
    xif6_0 = keras.layers.Conv2D(
        1504, (1, 1), strides=(1, 1))(xif5_4)
    xif6_0 = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.003, trainable=True)(xif6_0)
    xif6_0 = keras.layers.Activation("relu")(xif6_0)

    backbone = keras.layers.Dropout(0.2)(xif6_0)

    head = keras.layers.AveragePooling2D(pool_size=(1, 1))(backbone)
    head = keras.layers.Conv2D(
        1000, (1, 1), strides=(1, 1))(head)

    model = keras.Model(inputs=image_input, outputs=head)

    return model


class PytorchToKeras(object):
    def __init__(self, pModel, kModel):
        super(PytorchToKeras, self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel
        tf.keras.backend.set_learning_phase(0)

    def __retrieve_k_layers(self):
        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):

        input = torch.randn(input_size)
        input = Variable(input.unsqueeze(0))
        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module, "weight"):
                    # print(module)
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self, input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i, (source_layer, target_layer) in enumerate(zip(self.__source_layers, self.__target_layers)):
            # print(source_layer)
            weight_size = len(source_layer.weight.data.size())
            transpose_dims = []
            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)
            if isinstance(source_layer, nn.Conv2d):
                transpose_dims = [2,3,1,0]
                self.kModel.layers[target_layer].set_weights([source_layer.weight.data.numpy(
                ).transpose(transpose_dims), source_layer.bias.data.numpy()])
            elif isinstance(source_layer, nn.BatchNorm2d):
                self.kModel.layers[target_layer].set_weights([source_layer.weight.data.numpy(), source_layer.bias.data.numpy(),
                                                              source_layer.running_mean.data.numpy(), source_layer.running_var.data.numpy()])

    def save_model(self, output_file):
        self.kModel.save(output_file)

    def save_weights(self, output_file):
        self.kModel.save_weights(output_file, save_format='h5')


fbnet_pretrained_model = fbnet("fbnet_a", pretrained=True)
print('pretrained model loaded! ')
# print('model = ')
# print(fbnet_pretrained_model)

# get keras model
keras_model = KerasNet(input_shape=(224, 224, 3))
print('keras model loaded! ')

# get pytorch model
pytorch_model = fbnet_pretrained_model

# transfer weights
converter = PytorchToKeras(pytorch_model, keras_model)
converter.convert((3, 224, 224))
print('convert completed!')

# #Save the converted keras model for later use
# converter.save_weights("keras.h5")
converter.save_model("my_keras_model.h5")
print('keras h5 model weights saved!')

# # convert keras model to tflite model
# converter = tf.contrib.lite.TocoConverter.from_keras_model_file(
#     "keras_model.h5")
# tflite_model = converter.convert()
# open("convert_model.tflite", "wb").write(tflite_model)
# print('tflite model saved!')



