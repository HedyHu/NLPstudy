#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Hedy Hu'

import os
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import ListedColormap
import math
from scipy import stats
import seaborn as sns
import scipy
from fitter import Fitter #A tool to fit data to many distributions and best one(s) https://pypi.org/project/fitter/
import datetime
import re
from sklearn import preprocessing, manifold

def plot_hist(trainFrame, testFrame, cal_method, varname, bin_num):
    x = np.array(trainFrame.groupby('buyer_admin_id').agg({varname:cal_method})[varname])
    x_p1 = np.percentile(x, 1)
    x_p99 = np.percentile(x, 99)
    x_test = np.array(testFrame.groupby('buyer_admin_id').agg({varname:cal_method})[varname])
    x_test_p1 = np.percentile(x, 1)
    x_test_p99 = np.percentile(x, 99)
    # mu = x.mean()
    # sigma = x.std()
    # x_new = [x_p1 if x_item <= x_p1 else x_p99 if x_item >= x_p99 else x_item for x_item in x]
    x_new = [x_item for x_item in x if x_item < x_p99]
    x_test_new = [x_test_item for x_test_item in x_test if x_test_item < x_test_p99]
    figure_name = data_path + '/histgram_' + varname + '_' + cal_method.__name__ + '.png'
    # fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 12))
    fig, axes = plt.subplots(2, 2, sharex='row', figsize=(12, 12))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    #411 累计概率分布
    plt.sca(ax1)
    n, bins, patches = plt.hist(x_new, bin_num, cumulative=True, normed=1, facecolor='blue',alpha=0.7)# alpha: 透明度, normed: 每个条状图的占比例比
    # y = mlab.normpdf(bins, mu, sigma)  #python2.7代码
    # y = scipy.stats.norm.cdf(bins, mu, sigma) #累计概率分布
    _loc, _scale = scipy.stats.chi2.fit_loc_scale(x_new, len(bins))
    y = scipy.stats.chi2.cdf(bins, df=len(bins), loc=_loc, scale=_scale)
    plt.plot(bins, y, 'red')
    text = 'train' + ' ' + varname + ' ' + cal_method.__name__ + ' (N = ' + str(len(x)) + ')'
    plt.title(text)
    # plt.xlabel(varname)
    plt.ylabel('cumulative probability')
    plt.sca(ax2)
    n_test, bins_test, patches_test = plt.hist(x_test_new, bin_num, cumulative=True, normed=1, facecolor='blue',alpha=0.7)
    _loc_test, _scale_test = scipy.stats.chi2.fit_loc_scale(x_test_new, len(bins_test))
    y_test = scipy.stats.chi2.cdf(bins_test, df=len(bins_test), loc=_loc_test, scale=_scale_test)
    plt.plot(bins_test, y_test, 'yellow')
    text = 'test' + ' ' + varname + ' ' + cal_method.__name__ + ' (N = ' + str(len(x)) + ')'
    plt.title(text)
    # plt.xlabel(varname)
    plt.ylabel('cumulative probability')
    # plt.scatter(X, Y)  # plot
    # plt.plot(X, Y) #line
    #421 频数分布
    plt.sca(ax3)
    sns.distplot(x_new, bins=bin_num, kde=False) #也可以选择要核函数
    plt.title('train')
    plt.xlabel(varname)
    plt.ylabel('freq')
    plt.sca(ax4)
    sns.distplot(x_test_new, bins=bin_num, kde=False)
    plt.title('test')
    plt.xlabel(varname)
    plt.ylabel('freq')
    plt.savefig(figure_name)
    plt.close()

def plot_tsne(trainFrame, testFrame, n_cluster):
    trainFrame['xx_flag'] = trainFrame.apply(lambda x: 1 if x['buyer_country_id'] == 'xx' else 0, axis=1)
    testFrame['xx_flag'] = testFrame.apply(lambda x:2, axis=1)
    tsne_columns = [x for x in trainFrame.columns if len(re.findall('id',x)) == 0 \
                    and len(re.findall('min_order_time',x)) == 0 and len(re.findall('max_order_time', x)) == 0] #one-hot
    X = trainFrame[tsne_columns]
    X_test = testFrame[tsne_columns]
    scaler = preprocessing.StandardScaler().fit(X) #标准化处理
    X_standard = scaler.transform(X)
    X_test_standard = scaler.transform(X_test)
    X_total_standard = np.row_stack((X_standard, X_test_standard))
    tsne = manifold.TSNE(n_components=n_cluster, random_state=0) #TSNE
    X_total_tsne = tsne.fit_transform(X_total_standard)
    df_tsne = pd.concat([pd.DataFrame(X_total_tsne), pd.Series(list(trainFrame['xx_flag']) + list(testFrame['xx_flag']), name='xx_flag')], ignore_index=True, axis=1)
    plt.figure()
    text = 'T-SNE' + ' ' + '@' + str(n_cluster) + ' ' + 'clusters'
    figure_name = data_path + '/' + text + '.png'
    plt.title(text)
    plt.scatter(df_tsne[0], df_tsne[1], c=df_tsne[2].values, cmap=ListedColormap(["blue", "red", "yellow"]), marker='.')
    plt.savefig(figure_name)
    plt.close()


if __name__ == '__main__':
    path = os.getcwd()
    data_path = '/'.join(path.split('/')[:-1]) + '/' + 'data'
    train_file = data_path + '/' + 'Antai_AE_round1_train_20190626.csv'
    test_file = data_path + '/' + 'Antai_AE_round1_test_20190626.csv'
    item_attr = data_path + '/' + 'Antai_AE_round1_item_attr_20190626.csv'
    #匹配
    df_item_attr = pd.read_csv(item_attr, sep=',')
    item_attr_set = set(df_item_attr['item_id'])
    df_train_file = pd.read_csv(train_file, sep=',')
    df_train_file['absent_flag'] = df_train_file.apply(lambda x: 1 if x['item_id'] not in item_attr_set else 0, axis=1)
    df_test_file = pd.read_csv(test_file, sep=',')
    df_test_file['absent_flag'] = df_test_file.apply(lambda x: 1 if x['item_id'] not in item_attr_set else 0, axis=1)
    set1 = set(df_train_file['item_id'])
    set2 = set(df_test_file['item_id'])
    set_diff = set2.difference(set1)
    #20190625 train 9966641:46164 test 125419:918 set diff 21504
    #20190626 train 12843064:25445 test 166158:674 set_diff 28488

    missing_price = df_item_attr['item_price'].median()
    df_train_all = pd.merge(df_train_file, df_item_attr, how='left', left_on='item_id', right_on='item_id')
    df_test_all = pd.merge(df_test_file, df_item_attr, how='left', left_on='item_id', right_on='item_id')
    df_train_all['item_price'] = df_train_all.apply(lambda x: x['item_price'] if x['item_price'] > 0 else missing_price, axis=1) #价格缺失值补均值
    df_test_all['item_price'] = df_test_all.apply(lambda x: x['item_price'] if x['item_price'] > 0 else missing_price, axis=1) #价格缺失值补均值
    df_train_all = df_train_all.sort_values(by=['buyer_admin_id','irank']).reset_index()
    df_test_all = df_test_all.sort_values(by=['buyer_admin_id','irank']).reset_index()

    #区分经销商和个人：新客跟老客，大客户跟小客户不能放在一起计算
    plot_hist(df_train_all, df_test_all, pd.Series.count, 'item_id', 50)
    plot_hist(df_train_all, df_test_all, pd.Series.mean, 'item_price', 50)
    plot_hist(df_train_all, df_test_all, pd.Series.sum, 'item_price', 50)
    plot_hist(df_train_all, df_test_all, pd.Series.nunique, 'item_id', 50)
    plot_hist(df_train_all, df_test_all, pd.Series.nunique, 'cate_id', 50)
    plot_hist(df_train_all, df_test_all, pd.Series.nunique, 'store_id', 50)

    #除了图形之外，加一下距离上次购买时间间隔
    train_X = pd.concat([df_train_all[['buyer_admin_id','item_id']].groupby('buyer_admin_id').agg({'item_id':pd.Series.count}),\
                         df_train_all[['buyer_admin_id','item_price']].groupby('buyer_admin_id').agg({'item_price':pd.Series.mean}),
                         df_train_all[['buyer_admin_id','item_price']].groupby('buyer_admin_id').agg({'item_price':pd.Series.sum}),
                         df_train_all[['buyer_admin_id','item_id']].groupby('buyer_admin_id').agg({'item_id':pd.Series.nunique}),
                         df_train_all[['buyer_admin_id','cate_id']].groupby('buyer_admin_id').agg({'cate_id':pd.Series.nunique}),
                         df_train_all[['buyer_admin_id','store_id']].groupby('buyer_admin_id').agg({'store_id':pd.Series.nunique}),
                         df_train_all[['buyer_admin_id','create_order_time']].groupby('buyer_admin_id').agg({'create_order_time':pd.Series.min}),
                         df_train_all[['buyer_admin_id','create_order_time']].groupby('buyer_admin_id').agg({'create_order_time':pd.Series.max})],\
                        ignore_index=False, axis=1)
    train_X.columns = ['item_unit_freq','item_price_mean','item_price_sum','item_nunique','cate_nunique','store_nunique','min_order_time','max_order_time']
    train_X['train_X.buyer_admin_id'] = train_X.index
    train_X['order_time_gap'] = train_X.apply(lambda x: ((datetime.datetime.strptime(x['max_order_time'],"%Y-%m-%d %H:%M:%S") - \
                                                         datetime.datetime.strptime(x['min_order_time'],"%Y-%m-%d %H:%M:%S"))/x['item_unit_freq']).total_seconds(), axis=1)
    # train_X_append1 = df_train_all[['buyer_admin_id','buyer_country_id']].drop_duplicates(keep='last') #居然有不只是一个国家的客户
    train_X_append = df_train_all[['buyer_admin_id', 'buyer_country_id']].drop_duplicates(['buyer_admin_id'], keep='last')
    train_X_final = pd.merge(train_X, train_X_append, how='left', left_on='train_X.buyer_admin_id', right_on='buyer_admin_id')
    train_X_sample = train_X_final.sample(frac=0.1)
    # train_X_sample.to_csv(data_path + '/' + 'Antai_AE_round1_train_desc_sample.csv', index=False)

    test_X = pd.concat([df_test_all[['buyer_admin_id', 'item_id']].groupby('buyer_admin_id').agg({'item_id': pd.Series.count}),\
                        df_test_all[['buyer_admin_id', 'item_price']].groupby('buyer_admin_id').agg({'item_price': pd.Series.mean}),
                        df_test_all[['buyer_admin_id', 'item_price']].groupby('buyer_admin_id').agg({'item_price': pd.Series.sum}),
                        df_test_all[['buyer_admin_id', 'item_id']].groupby('buyer_admin_id').agg({'item_id': pd.Series.nunique}),
                        df_test_all[['buyer_admin_id', 'cate_id']].groupby('buyer_admin_id').agg({'cate_id': pd.Series.nunique}),
                        df_test_all[['buyer_admin_id', 'store_id']].groupby('buyer_admin_id').agg({'store_id': pd.Series.nunique}),
                        df_test_all[['buyer_admin_id', 'create_order_time']].groupby('buyer_admin_id').agg({'create_order_time': pd.Series.min}),
                        df_test_all[['buyer_admin_id', 'create_order_time']].groupby('buyer_admin_id').agg({'create_order_time': pd.Series.max})],\
                        ignore_index=False, axis=1)
    test_X.columns = ['item_unit_freq','item_price_mean','item_price_sum','item_nunique','cate_nunique','store_nunique','min_order_time','max_order_time']
    test_X['test_X.buyer_admin_id'] = test_X.index
    test_X['order_time_gap'] = test_X.apply(lambda x: ((datetime.datetime.strptime(x['max_order_time'], "%Y-%m-%d %H:%M:%S") - \
                                                        datetime.datetime.strptime(x['min_order_time'], "%Y-%m-%d %H:%M:%S"))/x['item_unit_freq']).total_seconds(), axis=1)

    test_X_append = df_test_all[['buyer_admin_id', 'buyer_country_id']].drop_duplicates(['buyer_admin_id'], keep='last')
    test_X_final = pd.merge(test_X, test_X_append, how='left', left_on='test_X.buyer_admin_id', right_on='buyer_admin_id')
    # test_X_final.to_csv(data_path + '/' + 'Antai_AE_round1_test_desc_total.csv', index=False)

    plot_tsne(train_X_sample, test_X_final, 2)
    plot_hist(train_X_final, test_X_final, pd.Series.mean, 'order_time_gap', 50)

    #t-SNE和UMAP聚类（UMAP聚类没做，t-SNE的结论是xx国跟yy国不同，到底怎么个不同法，从描述性统计上基本看不出来
    #使用FM方法，没法考虑时间穿越，只能在特征上面体现这种特点了吧 train:0713~0831 test:0717~0831
    df_train_user_item = df_train_all.groupby(['buyer_admin_id','item_id','cate_id','store_id','item_price']).agg({'item_id':pd.Series.count})
    df_train_user_item.rename(columns={'item_id':'item_freq'}, inplace=True)
    df_train_user_item.head(50000).to_csv(data_path + '/' + 'Antai_AE_round1_df_train_user_item_subset.csv')
    df_train_user_item.to_csv(data_path + '/' + 'Antai_AE_round1_df_train_user_item_all.csv')
    df_test_user_item = df_test_all.groupby(['buyer_admin_id','item_id','cate_id','store_id','item_price']).agg({'item_id':pd.Series.count})
    df_test_user_item.rename(columns={'item_id': 'item_freq'}, inplace=True)
    df_test_user_item.to_csv(data_path + '/' + 'Antai_AE_round1_df_test_user_item_all.csv')
