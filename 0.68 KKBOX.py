#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:02:37 2017

@author: Egoist
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

train_path = os.path.expanduser('/Users/apple/Desktop/KKBOX/train.csv') #os.path查阅一下
test_path = os.path.expanduser('/Users/apple/Desktop/KKBOX/test.csv')
songs_path = os.path.expanduser('/Users/apple/Desktop/KKBOX/songs.csv')
members_path = os.path.expanduser('/Users/apple/Desktop/KKBOX/members.csv')
song_extra_path = os.path.expanduser('/Users/apple/Desktop/KKBOX/song_extra_info.csv')

train = pd.read_csv(train_path, dtype={'target' : np.uint8,}) #指定'target'这一列的数据类型，有啥用？
test = pd.read_csv(test_path)
songs = pd.read_csv(songs_path)
members = pd.read_csv(members_path,dtype={'bd' : np.uint8},parse_dates=['registration_init_time','expiration_date']) #解析此2列的值作为独立的日期列
songs_extra = pd.read_csv(song_extra_path)





train = train.merge(songs, on='song_id', how='left') #左连接，左侧train取全部，右侧songs取部分
test = test.merge(songs, on='song_id', how='left')

#把会员的time series的序列提取出来
members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int) #这句有点长啊

members['registration_year'] = members['registration_init_time'].dt.year #.dt是有这个对象？？
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1) #删除这一列

def isrc_to_year(isrc): #对歌曲的时间做预处理
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True) #bool, default False, If True, do operation inplace and return None.

train = train.merge(members, on='msno', how='left') #train和members合并（msno）
test = test.merge(members, on='msno', how='left')


train = train.merge(songs_extra, on = 'song_id', how = 'left') #train和songs_extra合并（song_id）
train.song_length.fillna(200000,inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')


test = test.merge(songs_extra, on = 'song_id', how = 'left')
test.song_length.fillna(200000,inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')




train['genre_ids'] = train['genre_ids'].astype('object')
test['genre_ids'] = test['genre_ids'].astype('object')
train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids'] = train['genre_ids'].astype('category')
test['genre_ids'] = test['genre_ids'].astype('category')




def lyricist_count(x): #歌词作者计数
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1 #出现分割符，则+1
    return sum(map(x.count, ['|', '/', '\\', ';']))  #这是什么意思

train['lyricist'] = train['lyricist'].astype('object')
test['lyricist'] = test['lyricist'].astype('object')
train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8) #训练集添加一列新的feature
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)
train['lyricist'] = train['lyricist'].astype('category')
test['lyricist'] = test['lyricist'].astype('category')





def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)


#歌手，作词，作曲家 全部转换为category型
train['artist_name'] = train['artist_name'].astype('category')
test['artist_name'] = test['artist_name'].astype('category')
train['composer'] = train['composer'].astype('category')
test['composer'] = test['composer'].astype('category')
train['lyricist'] = train['lyricist'].astype('category')
test['lyricist'] = test['lyricist'].astype('category')



# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()} #这个是怎么用的，字典记录的是：song_id，播放次数
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}
def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError: #这儿是什么异常
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0
    
train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64) #添加歌曲播放次数这一列
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)


# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()} #字典记录的是：artist_name，播放次数
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}
def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0

train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)

train['language'] = train['language'].astype('category')
test['language'] = test['language'].astype('category')
train['city'] = train['city'].astype('category')
test['city'] = test['city'].astype('category')

for col in train.columns:
    if train[col].dtype == object: #所有object类型全部转化为category类型
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


train.to_csv('train_1_eng',index = False) #这个是干嘛？
test.to_csv('test_1_eng',index = False)




'''@@@@@@@@@@@@@@@@@@@@@@@tool area@@@@@@@@@@@@@@@@@@@@@@@'''
for each in train.columns:
    print(str(each) + (25-len(each))*' ' + str(train[each].dtype)) #这个是在干嘛？
    

category_col = ['gender',
                'language',
                'lyricist',
                'composer',
                'artist_name',
                'genre_ids',
                'source_type',
                'source_screen_name',
                'source_system_tab',
                'song_id',
                'msno',
                'city',
                'song_length_class']


numerical_col = ['song_length',
                 'bd',#outlier tackle  异常值处理？
                 'registered_via',
                 'expiration_date',
                 'membership_days',
                 'registration_year',
                 'registration_month',
                 'registration_date',
                 'expiration_year',
                 'expiration_month',
                 'song_year',#fillna
                 'lyricists_count',
                 'composer_count',
                 #'is_featured',
                 #'artist_count',
                 #'artist_composer',
                 #'artist_composer_lyricist',
                 #'song_lang_boolean',
                 #'smaller_song',
                 #'genre_ids_count',
                 'count_song_played',
                 'count_artist_played']

corr = pd.Series(index = numerical_col)  #这个是干嘛？
for each in numerical_col:
    c = train[each].corr(train['target'])
    corr[each] = c
    
 



















































