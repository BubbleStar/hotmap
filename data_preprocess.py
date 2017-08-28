#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:35:14 2017

@author: zhangxingxing
"""
#TODO
#封装警情数据预处理方法，使得可以按类别取出以格子为基准的警情数据
import pandas as pd
import numpy as np
import datetime

#%%朝阳区范围
LEFT_DOWN_X = 116.3589
LEFT_DOWN_Y = 39.8194
RIGHT_UP_X = 116.6535
RIGHT_UP_Y = 40.0959

def clean_data(dataPath):
    """
    :param dataPath: str, 原始警情数据路径
    :return: DataFrame, 清洗后的警情数据
    """
    df = pd.read_csv(dataPath)
    df = df[['警情序号', '报警时间', '出警派出所', '警情类别3', '经度', '纬度']]
    # 过滤经纬度为空的数据
    df = df[(df['经度'] != 'null') & (df['经度'] != 'NULL')]
    df = df.dropna()
    df['经度'] = df['经度'].astype('float64')
    df['纬度'] = df['纬度'].astype('float64')

    df = df[(df['经度'] >= LEFT_DOWN_X) & (df['经度'] <= RIGHT_UP_X) &
                        (df['纬度'] >= LEFT_DOWN_Y) & (df['纬度'] <= RIGHT_UP_Y)]
    df.columns = ['case_id', 'dates', 'poli_sta', 'type', 'lng', 'lat']
    df.reset_index()
    return df


def trans_by_type(crime_df, casetype="all", save_data_path="data/all.csv"):
    """
    :param crime_df: DataFrame, cleaned crime data
    :param casetype: str, 需要计算的警情类型, 默认指所有警情
    :return: "date_str","grid_id","该格子案件数","外层格子案件数"格式的警情数据
    """
    if(casetype != "all"):
        crime_df = crime_df[crime_df["type"] == casetype]
    crime_df['dates'] = crime_df['dates'].values.astype('datetime64[D]')

    crime_df = crime_df.sort_values('dates')
    crime_df['date'] = crime_df['dates'].apply(lambda x: x.strftime("%Y%m%d"))
    crime_df['time'] = crime_df['dates'].apply(lambda x: x.strftime("%H:%M:%S"))
    crime_df['lng'] = crime_df['lng'].astype('float64')
    crime_df['lat'] = crime_df['lat'].astype('float64')

    date_range = crime_df.groupby('date').count().index
    date_count = crime_df.groupby('date').count()['case_id']

    result_df = pd.DataFrame(columns=["date", "grid_id", "count_inside", "count"])

    for i in date_range:
        count_today = date_count[i]
        dict = {}
        df_temp = crime_df[crime_df['date'] == i]
        for index, item in df_temp.iterrows():

            grid_id = np.floor((item['lng'] - 116.3589) / 0.002946) * 124 + np.ceil((item['lat'] - 39.8154) / 0.002244)

            if (grid_id in dict):
                dict[grid_id] = dict[grid_id] + 1
            else:
                dict[grid_id] = 1
        grid_arr = []
        count_inside_arr = []
        for key, value in dict.items():
            grid_arr.append(key)
            count_inside_arr.append(value)

        length = len(grid_arr)
        df_temp = pd.DataFrame()
        df_temp['date'] = [i] * length
        df_temp['grid_id'] = grid_arr
        df_temp['count_inside'] = count_inside_arr
        df_temp['count'] = [count_today] * length

        result_df = pd.concat([result_df, df_temp])

    result_df['count_outside'] = result_df['count'] - result_df['count_inside']
    result_df = result_df.drop(["count"], axis=1)
    # return result_df
    result_df.to_csv(save_data_path, index=None, header=False)