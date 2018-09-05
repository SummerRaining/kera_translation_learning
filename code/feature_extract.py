# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:06:50 2018

@author: tunan
"""

'''
自己总结一份transla learning加上每张图片的hog值
使用模型融合
data_gene.flow_from_direction
一、h5py存入字符串
   1、规定格式 dt =  h5py.special_dtype(vlen=str)
   2、讲字符串列表转成字符串array
   3、创建h5py文件，创建数组ds = h.creat_dataset('a',a.shape,dtype = dt),注意直接第二个参数用a.shape
       写入数据 ds[:] = a
   4、读出 np.array(h['a'])
'''
from keras.layers import GlobalAveragePooling2D
from keras.applications import vgg16,inception_v3,resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import h5py
from tqdm import tqdm
import os,cv2

def get_test(Models,lamda_func):
    #得到test的特征和文件名
    path = r"C:\Users\tunan\Desktop\xuelang\resize_test"
    basic_model = Models(include_top = False,weights = 'imagenet')
    feature = GlobalAveragePooling2D()(basic_model.output)
    model = Model(inputs = basic_model.input,outputs = feature)
    file_name = os.listdir(path)
    path_name = [os.path.join(path,x) for x in file_name]
    test_data = []
    for p in tqdm(path_name):
        data = cv2.imread(p)
        data = np.expand_dims(data,axis = 0)
        data = lamda_func(data)
        feature = model.predict(data)
        test_data.append(feature)
    return np.concatenate(test_data,axis = 0),np.array(file_name)

def white_gap(input_size,Models,lamda_func,name):
    basic_model = Models(include_top = False,weights = 'imagenet')
    feature = GlobalAveragePooling2D()(basic_model.output)
    model = Model(inputs = basic_model.input,outputs = feature)
    gen = ImageDataGenerator()
    train_gen = gen.flow_from_directory(r"data\valAndTrain\train",
                                        shuffle = False,batch_size = 16,
                                        class_mode = 'categorical',
                                        target_size = input_size,
                                        interpolation='antialias',
                                         )
    val_gen = gen.flow_from_directory(r"data\valAndTrain\val",
                                        shuffle = False,batch_size = 16,
                                        class_mode = 'categorical',
                                        target_size = input_size,
                                        interpolation='antialias',
                                        )
    #predict_generator中的steps是指预测多少个batch，即生成器生成多少次。train_gen.n是所有样本数
    ntrain_batch = int(np.ceil(train_gen.n/16.))
    train_feature = []
    train_labels = []
    #将hog值和特征一批一批的提取出来
    for e in tqdm(range(ntrain_batch)):
        (data,label) = train_gen.next()
        train_labels.append(label)
        data =lamda_func(data)
        feature = model.predict(data)
        #提取特征，加入train_feature中
        train_feature.append(feature)
    train_feature = np.concatenate(train_feature,axis = 0)
    train_labels = np.concatenate(train_labels)
    #nval_batch是val上batch的次数，用一个train_feature能指定内存，无需动态分配
    nval_batch = int(np.ceil(val_gen.n/16.))
    val_dafeature = []
    val_labels = []
    for e in tqdm(range(nval_batch)):
        (data,label) = val_gen.next()
        val_labels.append(label)
        data =lamda_func(data)
        feature = model.predict(data)
        val_feature.append(feature)
    val_feature = np.concatenate(val_feature,axis = 0)
    val_labels = np.concatenate(val_labels)
    #得到test的特征
#    test_feature,test_name = get_test(Models,lamda_func)
    '''
    将数据以array的形式储存到h5文件中
    '''
#    dt = h5py.special_dtype(vlen = str)
    with h5py.File("data\\multi_gap_%s.h5"%(name)) as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("val", data=val_feature)
#        h.create_dataset('test',data = test_feature)
        h.create_dataset("train_labels", data=train_labels)
        h.create_dataset("val_labels", data=val_labels)
#        ds = h.create_dataset('file_name',test_name.shape,dtype = dt)
#        ds[:] = test_name
#使用vgg16,inception-v3,xception提取特征
os.chdir("..")
white_gap((600,600),vgg16.VGG16,vgg16.preprocess_input,'vgg16')
white_gap((600,600),inception_v3.InceptionV3,inception_v3.preprocess_input,'inception_v3')
white_gap((600,600),resnet50.ResNet50,resnet50.preprocess_input,'resnet50')