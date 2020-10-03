#!/usr/bin/env python3

# test keras model
from keras.models import load_model
from keras.utils import plot_model
#test tflite model
import numpy as np
import tensorflow as tf
import cv2


# test pretrained pytorch model
# from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from torchvision import models
import keras


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





# get the label
label = []
with open('val.txt','r') as file:
    cnt = 0
    for line in file.readlines():
        cnt = cnt+1
        i = 29
        tmp_str = ''
        while(line[i]!='\n'):
            tmp_str += line[i]
            i = i+1
        # print(tmp_str)
        label.append(int(tmp_str))
        if(cnt>=50000):
            break
print('label read done! ')
# print('label = ',label)

model = KerasNet()
# keras.utils.plot_model(model, show_shapes=False)
model.load_weights("./my_keras_model.h5")

print('keras model: ')
print(model.summary())
# model = fbnet("fbnet_a", pretrained=True) # 47%
# model = models.resnet50(pretrained=True) # 57%

total = 1
hit = 0.
for i in range(1,total+1):
    # get the input image
    if(i<10):
        test_path = './test_image/'+'ILSVRC2012_val_0000000'+str(i)+'.JPEG'
    elif(i<100):
        test_path = './test_image/'+'ILSVRC2012_val_000000'+str(i)+'.JPEG'
    elif(i<1000):
        test_path = './test_image/'+'ILSVRC2012_val_00000'+str(i)+'.JPEG'
    elif(i<10000):
        test_path = './test_image/'+'ILSVRC2012_val_0000'+str(i)+'.JPEG'
    else:
        test_path = './test_image/'+'ILSVRC2012_val_000'+str(i)+'.JPEG'
    out_path = './test_image/'+'test'+str(i)+'.JPEG'

    img = Image.open(test_path)
    transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
    ])
    img = transform1(img)
    img = np.asarray(img)
    # img.save(out_path) # save the cropped img
    # print('img old shape = ',img.shape)# [224, 224, 3]
    try: # RGB situation
        trash = img.shape[2]
    except:
        img = img[:,:,np.newaxis]
        img0 = img
        img = np.concatenate((img,img0),axis=2)
        img = np.concatenate((img,img0),axis=2)
    img = img[np.newaxis,:]
    img = img.astype(np.float32)
    img = img/255
    # print('img type = ',type(img))# tensorflow.python.framework.ops.EagerTensor
    print('img shape = ',img.shape)# [1, 224, 224, 3]
    # print('img dtype = ',img.dtype)# dtype: 'float32'
    print('here!!!')
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    GPU_list = tf.config.experimental.list_physical_devices('XLA_GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))
    config = tf.compat.v1.ConfigProto( device_count = {'XLA_GPU': 0 , 'XLA_GPU': 1} ) 
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # output = model(img)
    with tf.compat.v1.Session() as sess:
        sess.run(model(img))
        # print('output = ',output)
        # # predictions = model.predict(img)
        # print ('output shape = ', output.shape)# [1, 1000]
        # _, predicted = torch.max(output.data, 1)
        # print('predicted = ',predicted)
        # if(predicted.data == label[i-1]):
        #     hit += 1
        # out_print = 'ILSVRC2012_val_0000000'+str(i)+'.JPEG'
        # print(out_print,end=' ')
        # print('%d \tacc = %.2f' % (predicted.data,hit/i))


# with K.get_session() as sess:



# [name: "/device:CPU:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 18349790022318270149
# , name: "/device:XLA_CPU:0"
# device_type: "XLA_CPU"
# memory_limit: 17179869184
# locality {
# }
# incarnation: 17892500198425873053
# physical_device_desc: "device: XLA_CPU device"
# , name: "/device:XLA_GPU:0"
# device_type: "XLA_GPU"
# memory_limit: 17179869184
# locality {
# }
# incarnation: 12433492577318074615
# physical_device_desc: "device: XLA_GPU device"
# , name: "/device:XLA_GPU:1"
# device_type: "XLA_GPU"
# memory_limit: 17179869184
# locality {
# }
# incarnation: 2705339757748930799
# physical_device_desc: "device: XLA_GPU device"
# ]



# image = cv2.imread('./test_image/1.JPEG')
# img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # RGB图像转为gray

#  #需要用reshape定义出例子的个数，图片的 通道数，图片的长与宽。具体的参加keras文档
# img = (img.reshape(1, 224, 224, 3)).astype('int32')/255 
# predict = model.predict_classes(img)
# print ('predict: ')
# print (predict)

# cv2.imshow("Image1", image)
# cv2.waitKey(0)


# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="./my_keras_model_optimize_float16.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print(input_details)
# print(output_details)






