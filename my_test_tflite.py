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


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./my_keras_model_optimize_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)








# # get the label
# label = []
# with open('val.txt','r') as file:
#     cnt = 0
#     for line in file.readlines():
#         cnt = cnt+1
#         i = 29
#         tmp_str = ''
#         while(line[i]!='\n'):
#             tmp_str += line[i]
#             i = i+1
#         # print(tmp_str)
#         label.append(int(tmp_str))
#         if(cnt>=50000):
#             break
# print('label read done! ')
# # print('label = ',label)



# # model = fbnet("fbnet_a", pretrained=True) # 47%
# # model = models.resnet50(pretrained=True) # 57%

# total = 50000
# hit = 0.
# for i in range(1,total+1):
#     # get the input image
#     if(i<10):
#         test_path = './test_image/'+'ILSVRC2012_val_0000000'+str(i)+'.JPEG'
#     elif(i<100):
#         test_path = './test_image/'+'ILSVRC2012_val_000000'+str(i)+'.JPEG'
#     elif(i<1000):
#         test_path = './test_image/'+'ILSVRC2012_val_00000'+str(i)+'.JPEG'
#     elif(i<10000):
#         test_path = './test_image/'+'ILSVRC2012_val_0000'+str(i)+'.JPEG'
#     else:
#         test_path = './test_image/'+'ILSVRC2012_val_000'+str(i)+'.JPEG'
#     out_path = './test_image/'+'test'+str(i)+'.JPEG'

#     img = Image.open(test_path)
#     transform1 = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop((224, 224)),
#     ])
#     img = transform1(img)
#     img = np.asarray(img)
#     # img.save(out_path) # save the cropped img
#     # print('img old shape = ',img.shape)# [224, 224, 3]
#     try: # RGB situation
#         trash = img.shape[2]
#     except:
#         img = img[:,:,np.newaxis]
#         img0 = img
#         img = np.concatenate((img,img0),axis=2)
#         img = np.concatenate((img,img0),axis=2)
#     img = img[np.newaxis,:]
#     img = img.astype(np.float32)
#     img = img/255
#     print('img type = ',type(img))# tensorflow.python.framework.ops.EagerTensor
#     print('img shape = ',img.shape)# [1, 224, 224, 3]
#     print('img dtype = ',img.dtype)# dtype: 'float32'
#     output = model(img)
#     # print(output)
#     # print ('output shape = ', output.shape)# [1, 1000]
#     _, predicted = torch.max(output.data, 1)
#     if(predicted.data == label[i-1]):
#         hit += 1
#     out_print = 'ILSVRC2012_val_0000000'+str(i)+'.JPEG'
#     print(out_print,end=' ')
#     print('%d \tacc = %.2f' % (predicted.data,hit/i))












