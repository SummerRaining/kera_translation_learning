# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:45:49 2018

@author: tunan
"""
'''
一、添加正则项：
from keras import regularizers
可以在Dense(),Conv2d上添加 kernel_regularizer=regularizers.l2(0.01)
二、二分类时，最后一层的大小是全连接到1，acitivation是sigmoid
Dense(1, activation='sigmoid')
三、tdqm函数 from tfqm import tqdm
'''
import os
import glob
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adagrad
from keras import regularizers
from keras.applications.vgg16 import VGG16
#1、将训练集和测试集分开，resize到299*299
#2、调整读取文件数的函数
def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
  
# 数据准备
IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3指定的图片尺寸
FC_SIZE = 512                # 全连接层的节点个数
train_dir = "C:\\Users\\tunan\\Desktop\\inception-V3-translation\\valAndTrain\\train"  # 训练集数据
val_dir = "C:\\Users\\tunan\\Desktop\\inception-V3-translation\\valAndTrain\\val" # 验证集数据
nb_classes= 5
nb_epoch = 30
batch_size = 32

nb_epoch = int(nb_epoch)                # epoch数量
batch_size = int(batch_size)           

#　图片生成器,inception_v3与vgg不同,vgg只用缩小到/255
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,class_mode='categorical')
# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax',
                      kernel_regularizer = regularizers.l2(0.001))(x) #new softmax layer
  model = Model(inputs=base_model.input, output=predictions)
  return model
# 定义网络框架
base_model = VGG16(weights = 'imagenet',include_top = False)
model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层
# 冻上base_model所有层，这样就可以正确获得bottleneck特征
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
# 模式一训练
history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=100,
        class_weight='auto')
model.save("vgg16.h5")
