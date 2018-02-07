#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:43:11 2017

@author: Egoist
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import random
from collections import Counter 

from sklearn import metrics



print('start reading data')

train = pd.read_csv('name_train.csv')
test = pd.read_csv('name_test.csv')
df = pd.concat([train,test]) #这个是在干嘛？合并训练与测试集？

#为何要去掉id, target
train = df[0:len(train)].drop('id',axis = 1) 
test = df[len(train):].drop('target',axis = 1)
print('reading data finished')
l = pd.read_csv('better_train.csv') #这个文件是干嘛的
col = list(train.columns) #训练集的所有特征名
col1 = list(test.columns) #测试机的所有特征名

#为何要去掉use？
train = train.drop('use',axis = 1)
test = test.drop('use',axis = 1)

#以下特征全删除
drop = ['is_featured','genre_ids_count','artist_count','artist_composer','artist_composer_lyricist','song_lang_boolean','smaller_song',]
for each in drop:
    train = train.drop(each,axis = 1)
    test = test.drop(each, axis = 1)


#年龄异常值的处理
def age_transfer(age):#deal with bd's outlier
    new_age = age
    if age == 0:
        new_age = random.randint(7,50)
    if age <= 7 and age > 0:
        new_age = random.randint(12,25)
    if age >= 75:
        new_age = random.randint(45,75)
    return new_age

train['bd'] = train['bd'].apply(age_transfer).astype(np.int64)
test['bd'] = test['bd'].apply(age_transfer).astype(np.int64)

train['song_year'] = train['song_year'].fillna(train['song_year'].median()).astype('int64')
test['song_year'] = test['song_year'].fillna(test['song_year'].median()).astype('int64')

train['source_screen_name'] = train['source_screen_name'].fillna('Unknown')
test['source_screen_name'] = test['source_screen_name'].fillna('Unknown')

train['source_type'] = train['source_type'].fillna('Unknown')
test['source_type'] = test['source_type'].fillna('Unknown')

train['source_system_tab'] = train['source_system_tab'].fillna('Unknown')
test['source_system_tab'] = test['source_system_tab'].fillna('Unknown')


#new features


'''$$$$$$$$$$$$$$$$$$$'''
df = [train,test]
for each in df:
    each['use'] = 1 #这个又什么用

    print('start making dics......')
    # Calculate every member’s number of play behaviors
    m_c = dict(each['use'].groupby(each['msno']).sum())
    print('m_c finished')

    # Calculate every member’s number of play behaviors of each artist
    m_a_c = dict(each['use'].groupby([each['msno'],each['artist_name']]).sum())
    print('m_a_c finished')

    # Calculate every member’s number of play behaviors of each source_screen_name
    m_s_c = dict(each['use'].groupby([each['msno'],each['source_screen_name']]).sum())
    print('m_s_c finished')

    # Calculate every member’s number of play behaviors of each source_type
    m_st_c = dict(each['use'].groupby([each['msno'],each['source_type']]).sum())
    print('m_st_c finished')

    # Calculate every member’s number of play behaviors of each composer
    m_c_c = dict(each['use'].groupby([each['msno'],each['composer']]).sum())
    print('m_c_c finished')

    # Calculate every member’s number of play behaviors of each source_screen_name 
    m_a_s_c = dict(each['use'].groupby([each['msno'],each['artist_name'],each['source_type']]).sum())
    print('m_a_s_c finished')

    print('dics making finished')

    m_c_d = []
    m_a_c_d = []
    m_s_c_d = []
    m_st_c_d = []
    m_c_c_d = []
    m_a_s_c_d = []

    print('start iterrows......') #这个是啥？访问矩阵元素？
    for i,row in each.iterrows():
        m_c_d.append(m_c[row['msno']])

        tup = (row['msno'],row['artist_name']) #元组
        m_a_c_d.append(m_a_c[tup])

        tup = (row['msno'],row['source_screen_name'])
        m_s_c_d.append(m_s_c[tup])

        tup = (row['msno'],row['source_type'])
        m_st_c_d.append(m_st_c[tup])

        tup = (row['msno'],row['composer'])
        m_c_c_d.append(m_c_c[tup])

        tup = (row['msno'],row['artist_name'],row['source_type'])
        m_a_s_c_d.append(m_a_s_c[tup])
    print('iterrows finished and start adding to df......')

    #把新构造的特征加到特征矩阵中
    each['m_c'] = m_c_d
    each['m_a_c'] = m_a_c_d
    each['m_s_c'] = m_s_c_d
    each['m_st_c'] = m_st_c_d
    each['m_c_c'] = m_c_c_d
    each['m_a_s_c'] = m_a_s_c_d

    print('adding finished and start calculating ratios') 
    each['m_a_c_ratio'] = (each['m_a_c'] / each['m_c']) * 100    
    each['m_s_c_ratio'] = (each['m_s_c'] / each['m_c']) * 100
    each['m_st_c_ratio'] = (each['m_st_c'] / each['m_c']) * 100
    each['m_c_c_ratio'] = (each['m_c_c'] / each['m_c']) * 100
    each['m_a_s_c_ratio'] = (each['m_a_s_c'] / each['m_c']) * 100
    each = each.drop('use',axis = 1)   #删除‘use’这一列
    print('finished') 
'''$$$$$$$$$$$$$$$$$$$'''

#about love，这都行？？厉害
def is_love(song_name):
    name = str(song_name)
    if ('爱' in name or 'love' in name):
        is_love = 1
    else:
        is_love = 0
    return is_love

train['is_love'] = train['name'].apply(is_love).astype('bool')
test['is_love'] = test['name'].apply(is_love).astype('bool')

#genre_id
Counter(train['genre_ids']) #counter是干嘛的？
train['name'] = train['name'].fillna('Unknown')
test['name'] = test['name'].fillna('Unknown')
df = pd.concat([train,test])
k = [train,test]
for each in k:
    each['use'] = 1
    print('start making dics......')
    m_n_c = dict(each['use'].groupby([each['msno'],each['name']]).sum())
    print('m_n_c finished')
    m_g_c = dict(each['use'].groupby([each['msno'],each['genre_ids']]).sum())
    print('m_g_c finished')
    print('dics making finished')
    m_n_c_d = []
    m_g_c_d = []
    print('start iterrows......')
    for i,row in each.iterrows():
        tup = (row['msno'],row['name'])
        m_n_c_d.append(m_n_c[tup])
        tup = (row['msno'],row['genre_ids'])
        m_g_c_d.append(m_g_c[tup])
    print('iterrows finished and start adding to df......')
    each['m_n_c'] = m_n_c_d
    each['m_g_c'] = m_g_c_d
    print('adding finished and start calculating ratios') 
    each['m_n_c_ratio'] = (each['m_n_c'] / each['m_c']) * 100    
    each['m_g_c_ratio'] = (each['m_g_c'] / each['m_c']) * 100
    print('finished') 
train = train.drop('use',axis = 1)
test = test.drop('use',axis = 1)
train = train.drop('is_love',axis = 1)
test = test.drop('is_love',axis = 1)


#save，在这儿存的修改特征后的数据集。。。
train.to_csv('name_train.csv',index = False)
test.to_csv('name_test.csv',index = False)

#new features over




#model_train_area
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


# 本地评测
def local_test(train,test,num_rounds,learn_rate):
    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
    #divided validation
    train_val = train.tail(int(len(train)*0.2)) #用于测试的数据集
    train_trn = train.head(len(train)- int(len(train)*0.2)) #用于训练的数据集

    #make datasets
    x_local = train_trn.drop('target',axis = 1) #训练集特征矩阵
    y_local = train_trn['target'].values #训练集标签

    params_gdbt = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': learn_rate ,# 关键参数
        'verbose': 5,
        'num_leaves': 128, # 关键参数
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_seed': 2017,
        'feature_fraction': 0.9, # 关键参数
        'feature_fraction_seed': 2017,
        'max_bin': 512,
        'max_depth': 16, # 关键参数
        'num_rounds': num_rounds, #关键参数，太大会过拟合（不要超过1500）
        'metric' : 'auc'
    }

    d_train = lgb.Dataset(x_local,y_local)
    watchlist = lgb.Dataset(x_local,y_local) #这个有什么用？

    #model training
    print('start model training')
    model_gdbt_local = lgb.train(params_gdbt, train_set=d_train,  valid_sets=watchlist, verbose_eval=5) #保存训练后的模型
    print('model training finished')

    local_result = model_gdbt_local.predict(train_val.drop('target',axis = 1)) #本地划分的测试集预测的标签结果
    local_score = metrics.roc_auc_score(train_val['target'], local_result) #真实标签和预测标签对比，算出AUC，AUC越接近0.84，线上评测效果越好
    print(local_score)
    return local_score,model_gdbt_local


# 线上评测
def online_test(train,test,num_rounds,boost):
    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')

    #make datasets
    x_online = train.drop('target',axis = 1)
    y_online = train['target'].values #为何还要用values？
    params_gdbt = {
        'objective': 'binary',
        'boosting': boost,
        'learning_rate': 0.05 ,
        'verbose': 0,
        'num_leaves': 128,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_seed': 2017,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 2017,
        'max_bin': 512,
        'max_depth': 16,
        'num_rounds': num_rounds,
        'metric' : 'auc'
    }

    d_train = lgb.Dataset(x_online,y_online)
    watchlist = lgb.Dataset(x_online,y_online)

    #model training
    print('start model training')
    model_gdbt_online = lgb.train(params_gdbt, train_set=d_train,  valid_sets=watchlist, verbose_eval=5) #保存训练后的模型
    print('model training finished')

    print('strat predicting')
    online_result = model_gdbt_online.predict(test.drop('id',axis = 1)) #线上测试集是要去掉‘id’这一列
    print('predicting ended')

    return online_result,model_gdbt_online #返回预测标签，训练模型

#存储基于线上测试集预测的标签
def output(online_result):
    print('start outputing...')
    subm = pd.DataFrame()
    subm['id'] = test['id'].astype('int32').values
    subm['target'] = online_result
    subm.to_csv('/Users/apple/Desktop/submission_online.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
    print('outputing ended')

#这个有点6666啊，可以看特征的重要性？
def importance(model):
    feature_importance = pd.Series(index = model.feature_name(),data = model.feature_importance())
    return feature_importance




local_score,model_local = local_test(train,test,100,0.05)
online_result_gbdt,model_online_gbdt = online_test(train,test,1500,'gbdt')

online_result_dart,model_online_dart = online_test(train,test,1500,'dart') #‘dart’是啥？

output(online_result_dart)

new_result['target'] = f_result #这个是啥？？？最后的结果？
new_result.to_csv('/Users/apple/Desktop/submission_online.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
    
feature_importance = importance(model_online_gbdt)


model_online.save_model('0.70095') #这个写错了？？应该是model_online_gbdt？
model_online_dart.save_model('dart_0.70096')


model1 = lgb.Booster(model_file = '/Users/apple/Desktop/KKBOX/models/0.70095')
model2 = lgb.Booster(model_file = '/Users/apple/Desktop/KKBOX/models/dart')

f_result = model1.predict(test.drop('id',axis = 1))
dart_result = model2.predict(test.drop('id',axis = 1))
final_result = (dart + gbdt + new)/3  #这个new是啥？？怎么是倒着来的啊？
output(f_result)

new_result = pd.read_csv('0.70095')
dart = list(dart_result['target'])
gbdt = list(new_result['target'])
new = list(new_result['target'])
f_result = list(k['target'])
j_result = list(k['target'])
l_result = list(k['target'])
new_dart = list(online_result_dart)
f_result = list(range(len(dart)))

for i in range(len(dart)):
    f_result[i] = (gbdt[i] + new_dart[i])/2





## 下面这个tool area是干啥的？
'''@@@@@@@@@@@@@@@@@@@@@@@tool area@@@@@@@@@@@@@@@@@@@@@@@'''

k = Counter(train['song_id'].astype('object')+train['msno'].astype('object'))
Counter(train['source_screen_name'].fillna('Unknown'))
Counter(test['source_screen_name'].fillna('Unknown'))
source_screen_name = pd.get_dummies(train['source_screen_name'].fillna('Unknown'))
for each in train.columns:
    print(str(each) + (25-len(each))*' ' + str(train[each].dtype))
for df in [train,test]:
    print('------------')
    for each in df.columns:
        print(str(each) + (25-len(each))*' ' + str(df[each].isnull().values.any()))

Counter(train['song_id'])
train = k[0:len(train)]
test = k[len(train):]
train = train.drop('id',axis = 1)
test = test.drop('target',axis = 1)
train['count_composer_played']
col_best = list(model.feature_name())
'''drop features'''
train = train.drop('m_a_c_c',axis = 1)
test = test.drop('m_a_c_c',axis = 1)
train = train.drop('m_a_c_c ratio',axis = 1)
test = test.drop('m_a_c_c ratio',axis = 1)
train = train.drop('m_c_c',axis = 1)
test = test.drop('m_c_c',axis = 1)
drop = ['add','sub','mul','div']
for each in drop:
    train = train.drop(each,axis = 1)
    test = test.drop(each,axis = 1)
    
    
    
    
m = [train,test]
for each in m:
    each['use'] = 1
    s_c = dict(each['use'].groupby(each['song_id']).sum())
    print('s_c finished')
    song_s_c = dict(each['use'].groupby([each['song_id'],each['source_screen_name']]).sum())
    print('song_s_c finished')
    s_c_d = []
    song_s_c_d = []
    print('start iterrows...')
    for i,row in each.iterrows():
        s_c_d.append(s_c[row['song_id']])
        tup = (row['song_id'],row['source_screen_name'])
        song_s_c_d.append(song_s_c[tup])
    print('iterrows finished and start adding to df......')
    each['s_c'] = s_c_d
    each['song_s_c'] = song_s_c_d
    each['song_s_c_ratio'] = (each['song_s_c'] / each['s_c']) * 100
    each = each.drop('use',axis = 1)
    print('finished')


Counter(train['source_screen_name'])
Counter(train['source_type'])
Counter(train['source_system_tab'])    
train['name']
    
    
'''
    
    
    
    
    
#useless features
    
#count_composer_played
_dict_count_composer_played_train = {k: v for k, v in train['composer'].value_counts().iteritems()}
_dict_count_composer_played_test = {k: v for k, v in test['composer'].value_counts().iteritems()}
def count_composer_played(x):
    try:
        return _dict_count_composer_played_train[x]
    except KeyError:
        try:
            return _dict_count_composer_played_test[x]
        except KeyError:
            return 0
train['count_composer_played'] = train['composer'].apply(count_composer_played).astype(np.int64)
test['count_composer_played'] = test['composer'].apply(count_composer_played).astype(np.int64)

#artist_song_played
train['art_song'] = train['count_song_played'] * train['count_artist_played'] / (train['count_song_played'] + train['count_artist_played'])
test['art_song'] = test['count_song_played'] * test['count_artist_played'] / (test['count_song_played'] + test['count_artist_played'])
train = train.drop('art_song',axis = 1)
test = test.drop('art_song',axis = 1)

#MeanEncoder(msno)
new_df = pd.concat([train,test])
new_df["msno"] = new_df['msno'].astype('object')
train['msno'] = train['msno'].astype('object')
new_df["msno"] = new_df["msno"].fillna(new_df["msno"].mode()[0])
msno_new_df = pd.DataFrame(new_df['msno'], columns = ['msno'])
fit_pd = pd.DataFrame(train['msno'], columns = ['msno'])
meanencoder = MeanEncoder(categorical_features= ['msno'])
print('start computing MeanEncoder')
meanfeature_msno = meanencoder.fit_transform(fit_pd,train.target)
result_msno = meanencoder.transform(msno_new_df)
new_df['msno_mean'] = result_msno['msno_pred_0']
train['msno'] = train['msno'].astype('category')
train = new_df[0:len(train)].drop('id',axis = 1)
test = new_df[len(train):].drop('target',axis = 1)

#MeanEncoder(artist_name)
new_df = pd.concat([train,test])
new_df["artist_name"] = new_df['artist_name'].astype('object')
train['artist_name'] = train['artist_name'].astype('object')
new_df["artist_name"] = new_df["artist_name"].fillna(new_df["artist_name"].mode()[0])
artist_name_new_df = pd.DataFrame(new_df['artist_name'], columns = ['artist_name'])
fit_pd = pd.DataFrame(train['artist_name'], columns = ['artist_name'])
meanencoder = MeanEncoder(categorical_features= ['artist_name'])
print('start computing MeanEncoder')
meanfeature = meanencoder.fit_transform(fit_pd,train.target)
result_mean = meanencoder.transform(artist_name_new_df)
new_df['artist_name_mean'] = result_mean['artist_name_pred_0.0']
train['artist_name'] = train['artist_name'].astype('category')
train = new_df[0:len(train)].drop('id',axis = 1)
test = new_df[len(train):].drop('target',axis = 1)


#msno_mean * artist_name_mean
train['msno * artist'] = train['msno_mean'] * train['artist_name_mean']
test['msno * artist'] = test['msno_mean'] * test['artist_name_mean']







#m_a_c ratio
m = [train,test]
for each in m:
    each['use'] = 1
    msno_c = dict(each['use'].groupby(each['msno']).sum())
    m_a_c = dict(each['use'].groupby([each['msno'],each['artist_name']]).sum())
    m_a_c_d = []
    m_c_d = []
    for i in range(len(each)):
        tup = (each.loc[i,'msno'],each.loc[i,'artist_name'])
        m_a_c_d.append(m_a_c[tup])
    for i in range(len(each)):
        c = msno_c[each.loc[i,'msno']]
        m_c_d.append(c)
    each['m_c'] = m_c_d
    each['m_a_c'] = m_a_c_d
    each = each.drop('use',axis = 1)
    each['ratio'] = (each['m_a_c'] / each['m_c']) * 100
    
#m_s_c
train['source_screen_name'] = train['source_screen_name'].fillna('Unknown')
test['source_screen_name'] = test['source_screen_name'].fillna('Unknown')
each = pd.concat([train,test])

each['use'] = 1
m_s_d = []
m_s_c = dict(each['use'].groupby([each['msno'],each['source_screen_name']]).sum())
for i in range(len(each)):
    tup = (each.iloc[i,21],each.iloc[i,30])
    m_s_d.append(m_s_c[tup])
each['m_s_c'] = m_s_d
each = each.drop('use',axis = 1)
each['m_s_c ratio'] = (train['m_s_c'] / train['m_c']) * 100

train['m_s_c'] = m_s_d[0:len(train)]
test['m_s_c'] = m_s_d[len(train):]
train['m_s_c ratio'] = (train['m_s_c'] / train['m_c']) * 100
test['m_s_c ratio'] = (test['m_s_c'] / test['m_c']) * 100


#m_st_c
train['source_type'] = train['source_type'].astype('object').fillna('Unknown')
test['source_type'] = test['source_type'].astype('object').fillna('Unknown')
each = pd.concat([train,test])

each['use'] = 1
m_st_d = []
m_st_c = dict(each['use'].groupby([each['msno'],each['source_type']]).sum())
for i in range(len(each)):
    tup = (each.iloc[i,23],each.iloc[i,34])
    m_st_d.append(m_st_c[tup])


train['m_st_c'] = m_st_d[0:len(train)]
test['m_st_c'] = m_st_d[len(train):]
train['m_st_c ratio'] = (train['m_st_c'] / train['m_c']) * 100
test['m_st_c ratio'] = (test['m_st_c'] / test['m_c']) * 100

#m_a_s_c
train['source_type'] = train['source_type'].astype('object').fillna('Unknown')
test['source_type'] = test['source_type'].astype('object').fillna('Unknown')
train['artist_name'] = train['artist_name'].astype('object').fillna('Unknown')
test['artist_name'] = test['artist_name'].astype('object').fillna('Unknown')
each = pd.concat([train,test])

each['use'] = 1
m_a_s_c_d = []
m_a_s_c = dict(each['use'].groupby([each['msno'],each['artist_name'],each['source_type']]).sum())
for i in range(len(each)):
    tup = (each.iloc[i,25],each.iloc[i,0],each.iloc[i,36])
    m_a_s_c_d.append(m_a_s_c[tup])

train['m_a_s_c'] = m_a_s_c_d[0:len(train)]
test['m_a_s_c'] = m_a_s_c_d[len(train):]
train['m_a_s_c ratio'] = (train['m_a_s_c'] / train['m_c']) * 100
test['m_a_s_c ratio'] = (test['m_a_s_c'] / test['m_c']) * 100


#m_c_c ratio
m = [train,test]
for each in m:
    each['use'] = 1
    msno_c = dict(each['use'].groupby(each['msno']).sum())
    m_c_c = dict(each['use'].groupby([each['msno'],each['composer']]).sum())
    m_c_c_d = []
    m_c_d = []
    for i in range(len(each)):
        tup = (each.loc[i,'msno'],each.loc[i,'composer'])
        m_c_c_d.append(m_c_c[tup])
    for i in range(len(each)):
        c = msno_c[each.loc[i,'msno']]
        m_c_d.append(c)
    each['m_c'] = m_c_d
    each['m_c_c'] = m_c_c_d
    each = each.drop('use',axis = 1)
    each['m_c_ratio'] = (each['m_c_c'] / each['m_c']) * 100