import sys, os.path
sys.path.insert(0, os.path.abspath("."))
#%%
import sys
import tensorflow as tf
import scipy.optimize as opt
import numpy as np
import data_utils
import data_preprocess
data_in_path, model_path, start_date, end_date, date_range, top_n, data_out_path = data_utils.read_config(".")

crime_data, class_count = data_utils.data_input("data/入室盗窃.csv")


def hit_grid_rate(y_, y):
    """
    :param y_: np.array,取topN后的格子列表序列
    :param y: np.array,实际案发格子的列表序列
    :return: 格子命中率
    """
    y_pred = y_.tolist()
    y_no_nan = list(map(lambda x: [] if np.isnan(x).all() else x, y))

    days = 0
    hit_rate = []
    for index,item in enumerate(y_no_nan):
        if(item == []):
            continue
        else:
            days = days + 1
            count = 0
            for i in item:
                if i in y_pred[index]:
                    count = count + 1
            hit_rate.append(count/len(item))
    # print(len(hit_rate), days)
    # print(hit_rate)
    return np.array(hit_rate)

# crime_data["noclass_layer0"] = np.ones(len(crime_data))
# print(crime_data)
train_x_data = data_utils.data_train_x(start_date, end_date, date_range, crime_data.ix[:, [0, 1, 2]])
train_y_data = data_utils.data_train_y(start_date, end_date, date_range, crime_data.ix[:, [0, 1, 2]]).values

X_shape = train_x_data.shape
w = np.ones(X_shape[1])
Y_ = data_utils.y_func(train_x_data, w, top_n)
    # print(Y_)
# stat, loss, hit_rate = data_utils.sort_loss(Y_, train_y_data)
# print(i)
hit_rate = hit_grid_rate(Y_, train_y_data)
# print(train_y_data[0],train_y_data[1])
# print(hit_rate)
# print(stat, loss, np.mean(hit_rate))
# trueY = [4867.0,2304.0,1,7009.0,2321.0,6752.0,4888.0,3548.0,4476.0,6721.0]
# print(len(train_y_data[0]))
# print(trueY)
# count = 0
# for i in trueY:
#     if i in Y_[0]:
#         count = count+ 1
# rate = count/len(trueY)
# print(rate)
print(np.mean(hit_rate))

