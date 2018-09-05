# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:39:10 2018

@author: tunan
"""

import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from sklearn.metrics import roc_auc_score
#from train_with_feature import build_test
import os
os.chdir("..")
x_train,x_val,x_test =[],[],[]
model_list = ['inception_v3','vgg16','resnet50']
for modelName in model_list:
    with h5py.File("data\\multi_gap_%s.h5"%(modelName),'r') as h:
        x_train.append(np.array(h["train"]))
        y_train = np.array(h['train_labels'])
        x_val.append(np.array(h['val']))
        y_val = np.array(h['val_labels'])
# =============================================================================
#         x_test.append(np.array(h['test']))
#         Lname = np.array(h['file_name'])
# =============================================================================
x_train = np.concatenate(x_train,axis = 1)
x_val = np.concatenate(x_val,axis = 1)
#x_test = np.concatenate(x_test,axis = 1)
#特征提取出来后随机打乱    
shuffle(x_train,y_train)
shuffle(x_val,y_val)
#构造图结构，函数化结构
inputs = Input((x_train.shape[1],))
x = Dense(512,activation = 'relu',
          kernel_initializer = 'TruncatedNormal')(inputs)
x = Dropout(0.5)(inputs)
outputs = Dense(5,activation = 'sigmoid',
                kernel_initializer = 'TruncatedNormal')(x)
model = Model(inputs = inputs,outputs = outputs)
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
#开始训练
model.fit(x= x_train,y = y_train,validation_data = (x_val,y_val),verbose = 2,
          batch_size = 128,epochs = 30)
#计算最后模型的auc值
ypre_val = model.predict(x_val)
print('auc is :',roc_auc_score(y_val,ypre_val))
#导出预测文件
#ypre_test = model.predict(x_test)
#result = 1-ypre_test
#build_test(result.reshape(-1,).tolist(),Lname.tolist(),'result_multi_hog2.csv')
