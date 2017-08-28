#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:50:44 2017

@author: zhangxingxing
"""
#%%
import sys, os.path
sys.path.insert(0, os.path.abspath("."))
#%%
import sys
import tensorflow as tf
import scipy.optimize as opt
import numpy as np
import data_utils
import data_preprocess

casetypes = ["all", '入室盗窃', '拦路抢劫', '砸撬汽车盗窃', '诈骗', '盗窃机动车', '抢夺',
             '扒窃', '拎包盗窃', '盗窃非机动车', '强奸', '入室抢劫', '抢劫出租汽车司机财物', '抢劫机动车', '侵占',
             '伤害（致伤）', '故意损坏财物', '敲诈勒索', '绑架', '凶杀或伤害致死','寻衅滋事',
             '网络诈骗', '聚众哄抢', '非法剥夺人身自由']



data_in_path, model_path, start_date, end_date, date_range, top_n, data_out_path = data_utils.read_config(".")

data_all = data_preprocess.clean_data("data/crime_data.csv")

crime_grid_path = []

for i in casetypes:
    data_preprocess.trans_by_type(data_all,i,"data/"+i+".csv")
    crime_grid_path.append("data/"+i+".csv")


for i in crime_grid_path:
    # 加载数据
    crime_data, class_count = data_utils.data_input(i)
    train_x_data = data_utils.data_train_x(start_date, end_date, date_range, crime_data.ix[:, [0, 1, 2]])
    train_y_data = data_utils.data_train_y(start_date, end_date, date_range, crime_data.ix[:, [0, 1, 2]]).values

    X_shape = train_x_data.shape
    w = np.ones(X_shape[1])
    Y_ = data_utils.y_func(train_x_data, w, top_n)
    # print(Y_)
    hit_rate = data_utils.hit_grid_rate(Y_, train_y_data)
    print(i)
    print(np.mean(hit_rate))