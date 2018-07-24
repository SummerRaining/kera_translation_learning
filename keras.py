# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:52:30 2018

@author: tunan
"""
'''
一：model的方法中：最基本的是model.fit(x,y)这句直接执行程序
      model.fit_generator(generator,)使用生成器训练模型。
      step_per_epoch：生成器返回多少次算作一次epoch（猜测imageDataGenerator的会先返回一个所有原始数据，
                      然后返回随机生成的数据）
      epochs:整数，数据迭代的次数
      verbose:日志记录，0不输出，1输出进度条，2每个epoch输出一行
      validation_data:两种形式1.（val_x,val_y）的元组。2.generator()
      class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
二、GlobalAveragePooling2D()(x)
     对x施加全局池化：输入形如（samples，channels, rows，cols）的4D张量,输出形如(nb_samples, channels)的2D张量
三、keras.models.Model(inputs,outputs)使用输入和输出构建模型,称为函数式构建模型。
     outputs = Dense(100)(inputs)，将模型或者layer当做函数使用，用outputs和inputs构建模型
     outputs是通过inputs计算出来的。
四、使用inception-v3做迁移学习：
    1、导入inceptionV3模型，使用InceptionV3(weights = "imagenet',include_top = False)
    2、获得base_model()提取出来的特征。x = base_model.output
    3、用GlobalAveragePooling2D()(x)将x变成只有n_channel的一维向量。使用函数式构建模型。得到新的模型model。
    4、注意model是基于以前的模型base_model而来的改变base_model会使model随之发生改变，
       改变base_model中的所有layer的trainable属性，base_model.layers()会获得所有layer的列表
    5、model.compile()规定学习过程。optimizer,loss,metrics
    5、构建ImageDataGenerator(),训练生成器和测试生成器。开始训练model.fit_generator(generator)
'''

from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras import backend as K

train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#不需要数据增强，只用做预处理就可以了
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
        "C:\\Users\\tunan\\Desktop\\inception-V3-translation\\valAndTrain\\train",
        target_size=(299, 299),
        batch_size=32,
        class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
        "C:\\Users\\tunan\\Desktop\\inception-V3-translation\\valAndTrain\\val",
        target_size=(299, 299),
        batch_size=32,
        class_mode="categorical")

# create the base pre-trained model,得到预训练的模型了
base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])
# train the model on the new data for a few epochs
#steps_per_epoch生成器每返回2000次记作一个epoch,epochs共做50次epochs,validation_steps验证集的生成器返回的次数
model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=100)

# =============================================================================
# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.
# 
# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
# 
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True
# 
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# 
# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit_generator(...)
# =============================================================================
