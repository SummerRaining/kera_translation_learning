# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:20:14 2018

@author: tunan
"""

#from reshapeAndLogist import build_resize,read_img,build_test_resize
import os
import numpy as np
import shutil
def read_img_path(file_path):
    path = os.listdir(file_path)
    path = [os.path.join(file_path,p) for p in path]
    return path
def build_remove(source,target):
    if not os.path.exists(target):
        os.mkdir(target)
        print("build target direction")
    for image in source:
        shutil.copy(image,target+'\\'+image.split("\\")[-1])
        
if __name__ == '__main__':
    os.chdir("..")
    #分到train和test中
    SOURCE_PATH = 'data\\flower_photos'
    resizeTrainAndTest = "data\\valAndTrain"
    if not os.path.exists(resizeTrainAndTest):
        os.mkdir(resizeTrainAndTest)
        print("build resizeTrainAndTest direction")
    TRAIN_PATH = os.path.join(resizeTrainAndTest,"train")
    VAL_PATH = os.path.join(resizeTrainAndTest,"val")
    #切换到训练夹目录下后
    if not os.path.exists(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)
        print("build TRAIN_PATH direction")
    if not os.path.exists(VAL_PATH):
        os.mkdir(VAL_PATH)
        print("build TEST_PATH direction")
    for cur_path in os.listdir(SOURCE_PATH):
        cur =np.asarray(read_img_path(os.path.join(SOURCE_PATH,cur_path)))
        np.random.shuffle(cur)
        train = cur[:int(len(cur)*0.9)]
        val = cur[int(len(cur)*0.9):]
        #文件移动resizeTrainAndTest下的train和val中的norm和flaws
        build_remove(train,os.path.join(resizeTrainAndTest,'train',cur_path))
        build_remove(val,os.path.join(resizeTrainAndTest,'val',cur_path))
