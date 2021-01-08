#!/usr/bin/env python
# coding: utf-8

from lightgbm import LGBMClassifier
import pandas as pd
import pandas_profiling
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from joblib import Parallel,delayed
import numpy as np
import re
import pandas_profiling
import time
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn.utils import shuffle
import catboost
from catboost import CatBoostClassifier, Pool, cv
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



def time_transform(t):
    # 先转换为时间数组
    timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    # 转换为时间戳
    timeStamp = int(time.mktime(timeArray))
    return timeStamp
def Z_score(mean_, std_,x):
    return (x-mean_)/std_
def max_mean_std(data):
    return np.max(data), np.mean(data), np.std(data)
def calculate_acc(predictions, truth):
    hit = 0
    for i in range(len(predictions)):
        if predictions[i] == truth[i]:
            hit = hit +1
    return hit/len(predictions)


train_set = pd.read_csv('train_set.csv',converters={'label_list': eval})
test_set = pd.read_csv('test_set_course_vec.csv',converters={'label_list': eval, 'course_vecs_CNN':eval, 'course_vecs_LR':eval})
course_info = pd.read_json('course_info.json',lines=True)  #706 courses
video_info = pd.read_json('video_info.json',lines=True)   #38181 videos

videoID = video_info['id'].values.tolist()
courseID = course_info['course_id'].values.tolist()
videoID_encoder = LabelEncoder()
courseID_encoder = LabelEncoder()
videoID_encoder.fit(videoID)
courseID_encoder.fit(courseID)
course_info['courseID'] = course_info['course_id'].progress_apply(lambda x : courseID_encoder.transform([x]))
course_info['videoIDs'] = course_info['item'].progress_apply(lambda x : videoID_encoder.transform(x))


course_video_num = {}
def count_videos(courseId, videoIds):
    number_of_video = len(videoIds)
    course_video_num[courseId[0]] = number_of_video
    
course_info.progress_apply(lambda row: count_videos(row['courseID'],row['videoIDs']), axis=1)   



course_frequence= {}
frequence_list = []
course_ids = course_info['courseID'].values.tolist()
course_ids =[ item for elem in course_ids for item in elem]  #706 courses
for i in range(len(course_ids)):
    course_frequence[course_ids[i]]=0
# course_frequence[470]
def course_frequence_calculate(courseListIDs):
    courseListIDs=re.sub('\s+',  ' ',courseListIDs[1:-1].strip(' '))
    courses_ = [int(i) for i in courseListIDs.split(' ')]
    for i in range(len(courses_)):
        num = course_frequence[courses_[i]] +1
        course_frequence[courses_[i]] = num

train_set['courseListIDs'].progress_apply(lambda x : course_frequence_calculate(x))

for k,v in course_frequence.items():
    frequence_list.append(v)
    
mean_course_frequence, max_course_frequence, std_course_frequence = max_mean_std(frequence_list)


video_start_times_info = []
video_end_times_info = []
local_start_times_info = []
local_end_times_info = []
video_durations_info = []
local_watching_times_info = []
video_progress_times_info = []
watching_counts_info = []
local_interval_info = []

video_duration ={}

def collect_info(course_ids,video_ids,watching_counts,video_durations,local_watching_times,video_progress_times,
                    video_start_times, video_end_times, local_start_times,local_end_times,courseListIDs):
    course_ids = eval(course_ids)
    video_ids = eval(video_ids)
    watching_counts = eval(watching_counts)
    video_durations = eval(video_durations)
    local_watching_times = eval(local_watching_times)
    video_progress_times = eval(video_progress_times)
    video_start_times = eval(video_start_times)
    video_end_times = eval(video_end_times) 
    local_start_times = eval(local_start_times)
    local_end_times = eval(local_end_times)
   
    courseListIDs=re.sub('\s+',  ' ',courseListIDs[1:-1].strip(' '))
    courses_ = [int(i) for i in courseListIDs.split(' ')]
    courses = {}
    courses_textual_content = {}
    courses_textual_name = {}
    for i in range(len(courses_)):
        courses[courses_[i]] = []

    for i in range(len(course_ids)):
        course = course_ids[i]
        local_start_times_ = time_transform(local_start_times[i])
        local_end_times_ = time_transform(local_end_times[i])
        interval = local_end_times_ - local_start_times_

        video_start_times_info.append(video_start_times[i])
        video_end_times_info.append(video_end_times[i])
        local_start_times_info.append(local_start_times_)
        local_end_times_info.append(local_end_times_)
        video_durations_info.append(video_durations[i])
        
        video_duration[video_ids[i]] = video_durations[i]
        
        local_watching_times_info.append(local_watching_times[i])
        video_progress_times_info.append(video_progress_times[i])
        watching_counts_info.append(watching_counts[i])
        local_interval_info.append(interval)
            


train_set.progress_apply(lambda row: collect_info(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)            

    #global stastic info
std_local_start_times_info = np.std(local_start_times_info, ddof=1)
mean_local_start_times_info = np.mean(local_start_times_info)

std_video_start_times_info = np.std(video_start_times_info, ddof=1)
mean_video_start_times_info = np.mean(video_start_times_info)

std_video_end_times_info = np.std(video_end_times_info, ddof=1)
mean_video_end_times_info = np.mean(video_end_times_info)

std_local_end_times_info = np.std(local_end_times_info, ddof=1)
mean_local_end_times_info = np.mean(local_end_times_info)

std_video_durations_info = np.std(video_durations_info, ddof=1)
mean_video_durations_info = np.mean(video_durations_info)

std_local_watching_times_info = np.std(local_watching_times_info, ddof=1)
mean_local_watching_times_info = np.mean(local_watching_times_info)

std_video_progress_times_info = np.std(video_progress_times_info, ddof=1)
mean_video_progress_times_info = np.mean(video_progress_times_info)

std_watching_counts_info = np.std(watching_counts_info, ddof=1)
mean_watching_counts_info = np.mean(watching_counts_info)

std_local_interval_info = np.std(local_interval_info, ddof=1)
mean_local_interval_info = np.mean(local_interval_info)



def feature_genration(course_ids,video_ids,watching_counts,video_durations,local_watching_times,video_progress_times,
                    video_start_times, video_end_times, local_start_times,local_end_times,courseListIDs):
    course_ids = eval(course_ids)
    video_ids = eval(video_ids)
    watching_counts = eval(watching_counts)
    video_durations = eval(video_durations)
    local_watching_times = eval(local_watching_times)
    video_progress_times = eval(video_progress_times)
    video_start_times = eval(video_start_times)
    video_end_times = eval(video_end_times) 
    local_start_times = eval(local_start_times)
    local_end_times = eval(local_end_times)
    
    unix_start_time = [time_transform(i) for i in local_start_times]
    unix_end_time = [time_transform(i) for i in local_end_times]
    unix_interval = [unix_end_time[i] - unix_start_time[i] for i in range(len(unix_start_time))]
    
    z_score_local_start_time = [Z_score(mean_local_start_times_info,std_local_start_times_info,i) for i in unix_start_time]
    z_score_local_end_time = [Z_score(mean_local_end_times_info,std_local_end_times_info,i) for i in unix_end_time]
    z_score_interval = [Z_score(mean_local_interval_info,std_local_interval_info,i) for i in unix_interval]
    
    

    courseListIDs=re.sub('\s+',  ' ',courseListIDs[1:-1].strip(' '))
    courses_ = [int(i) for i in courseListIDs.split(' ')]
    courses = {}
 
    for i in range(len(courses_)):
        courses[courses_[i]] = []
#         courses_textual_name[courses_[i]] = []
#         courses_textual_content[courses_[i]] = []
    for i in range(len(course_ids)):
        course = course_ids[i]


        info_vec = [course_ids[i],watching_counts[i],video_durations[i],local_watching_times[i],video_progress_times[i],
                    video_start_times[i], video_end_times[i],z_score_interval[i],z_score_local_start_time[i],z_score_local_end_time[i]]   
    
        courses[course].append(info_vec)

    course_vec = []
    for i in range(len(courses_)):
        
        info_vecs = courses[courses_[i]]
        total_video_num = course_video_num[courses_[i]]
        num_of_vdo = len(info_vecs)
        percentage_viewed = num_of_vdo/total_video_num
        num_of_cos = len(courses_)
        course_watching_counts = [info_vecs[j][1] for j in range(len(info_vecs))]
        
        course_time_per_watching = [info_vecs[j][3]/info_vecs[j][1] for j in range(len(info_vecs))]
        
        course_video_durations = [info_vecs[j][2] for j in range(len(info_vecs))]
        course_local_watching_times = [info_vecs[j][3] for j in range(len(info_vecs))]
        course_video_progress_times = [info_vecs[j][4] for j in range(len(info_vecs))]
        course_video_watching_speed = [info_vecs[j][3]/info_vecs[j][4] for j in range(len(info_vecs))]
        course_video_start_times = [info_vecs[j][5] for j in range(len(info_vecs))]
        course_video_end_times = [info_vecs[j][6] for j in range(len(info_vecs))]
        course_z_score_interval = [info_vecs[j][7] for j in range(len(info_vecs))]
        course_z_score_local_start_time =[info_vecs[j][8] for j in range(len(info_vecs))]
        course_z_score_local_end_time =[info_vecs[j][9] for j in range(len(info_vecs))]
        
        
        max_watching_counts, mean_watching_counts, std_watching_counts = max_mean_std(course_watching_counts)
        max_video_durations, mean_video_durations, std_video_durations = max_mean_std(course_video_durations)
        max_local_watching_times, mean_local_watching_times, std_local_watching_times = max_mean_std(course_local_watching_times)
        max_video_progress_times, mean_video_progress_times, std_video_progress_times = max_mean_std(course_video_progress_times)
        max_video_watching_speed, mean_video_watching_speed, std_video_watching_speed = max_mean_std(course_video_watching_speed)
        max_video_start_times, mean_video_start_times, std_video_start_times = max_mean_std(course_video_start_times)
        max_video_end_times, mean_video_end_times, std_video_end_times = max_mean_std(course_video_end_times)
        max_z_score_interval, mean_z_score_interval, std_z_score_interval = max_mean_std(course_z_score_interval)
        max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time = max_mean_std(course_z_score_local_start_time)
        max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time = max_mean_std(course_z_score_local_end_time)
        
        max_time_per_watching, mean_time_per_watching, std_time_per_watching = max_mean_std(course_time_per_watching)
        
        #compared to global stastic
#         mean_watching_counts_ratio = mean_watching_counts/mean_watching_counts_info
#         mean_video_durations_ratio = mean_video_durations/mean_video_durations_info
#         mean_local_watching_times_ratio = mean_local_watching_times/mean_local_watching_times_info
#         mean_watching_speed_ratio = mean_video_watching_speed/(mean_local_watching_times_info/mean_video_progress_times_info)
#         std_watching_counts_ratio = std_watching_counts/std_watching_counts_info
#         std_video_durations_ratio = std_video_durations/std_video_durations_info
#         std_local_watching_times_ratio = std_local_watching_times/std_local_watching_times_info
#         std_watching_speed_ratio = std_video_watching_speed/(std_local_watching_times_info/std_video_progress_times_info)

        course_frequence_z_score = Z_score(mean_course_frequence,std_course_frequence,course_frequence[courses_[i]])
    
        total_watching_counts = np.sum(course_watching_counts)
        total_video_durations = np.sum(course_video_durations)
        total_local_watching_times = np.sum(course_local_watching_times)
        total_video_progress_times = np.sum(course_video_progress_times)
        
        vec = [courses_[i],num_of_cos,num_of_vdo,max_watching_counts, mean_watching_counts, std_watching_counts,
                max_video_durations, mean_video_durations, std_video_durations,
                max_local_watching_times, mean_local_watching_times, std_local_watching_times,
                max_video_progress_times, mean_video_progress_times, std_video_progress_times,
                max_video_watching_speed, mean_video_watching_speed, std_video_watching_speed, 
                max_video_start_times, mean_video_start_times, std_video_start_times,
                max_video_end_times, mean_video_end_times, std_video_end_times,
                max_z_score_interval, mean_z_score_interval, std_z_score_interval, 
                total_video_num, percentage_viewed,
                max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time,
                max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time,
#                 mean_watching_counts_ratio,mean_video_durations_ratio,mean_local_watching_times_ratio,mean_watching_speed_ratio,
#                 std_watching_counts_ratio,std_video_durations_ratio,std_local_watching_times_ratio,std_watching_speed_ratio
               course_frequence_z_score,max_time_per_watching, mean_time_per_watching, std_time_per_watching,
               total_watching_counts,total_video_durations,total_local_watching_times,total_video_progress_times
               
              ]
        
        course_vec.append(vec)

    
    return course_vec


def gbdt_prediction(course_ids,video_ids,watching_counts,video_durations,local_watching_times,video_progress_times,
                    video_start_times, video_end_times, local_start_times,local_end_times,courseListIDs):
    course_ids = eval(course_ids)
    video_ids = eval(video_ids)
    watching_counts = eval(watching_counts)
    video_durations = eval(video_durations)
    local_watching_times = eval(local_watching_times)
    video_progress_times = eval(video_progress_times)
    video_start_times = eval(video_start_times)
    video_end_times = eval(video_end_times) 
    local_start_times = eval(local_start_times)
    local_end_times = eval(local_end_times)
    
    unix_start_time = [time_transform(i) for i in local_start_times]
    unix_end_time = [time_transform(i) for i in local_end_times]
    unix_interval = [unix_end_time[i] - unix_start_time[i] for i in range(len(unix_start_time))]
    
    z_score_local_start_time = [Z_score(mean_local_start_times_info,std_local_start_times_info,i) for i in unix_start_time]
    z_score_local_end_time = [Z_score(mean_local_end_times_info,std_local_end_times_info,i) for i in unix_end_time]
    z_score_interval = [Z_score(mean_local_interval_info,std_local_interval_info,i) for i in unix_interval]
    
    

    courseListIDs=re.sub('\s+',  ' ',courseListIDs[1:-1].strip(' '))
    courses_ = [int(i) for i in courseListIDs.split(' ')]
    courses = {}
 
    for i in range(len(courses_)):
        courses[courses_[i]] = []
#         courses_textual_name[courses_[i]] = []
#         courses_textual_content[courses_[i]] = []
    for i in range(len(course_ids)):
        course = course_ids[i]


        info_vec = [course_ids[i],watching_counts[i],video_durations[i],local_watching_times[i],video_progress_times[i],
                    video_start_times[i], video_end_times[i],z_score_interval[i],z_score_local_start_time[i],z_score_local_end_time[i]]   
    
        courses[course].append(info_vec)

    course_vec = []
    for i in range(len(courses_)):
        
        info_vecs = courses[courses_[i]]
        total_video_num = course_video_num[courses_[i]]
        num_of_vdo = len(info_vecs)
        percentage_viewed = num_of_vdo/total_video_num
        num_of_cos = len(courses_)
        course_watching_counts = [info_vecs[j][1] for j in range(len(info_vecs))]
        
        course_time_per_watching = [info_vecs[j][3]/info_vecs[j][1] for j in range(len(info_vecs))]
        
        course_video_durations = [info_vecs[j][2] for j in range(len(info_vecs))]
        course_local_watching_times = [info_vecs[j][3] for j in range(len(info_vecs))]
        course_video_progress_times = [info_vecs[j][4] for j in range(len(info_vecs))]
        course_video_watching_speed = [info_vecs[j][3]/info_vecs[j][4] for j in range(len(info_vecs))]
        course_video_start_times = [info_vecs[j][5] for j in range(len(info_vecs))]
        course_video_end_times = [info_vecs[j][6] for j in range(len(info_vecs))]
        course_z_score_interval = [info_vecs[j][7] for j in range(len(info_vecs))]
        course_z_score_local_start_time =[info_vecs[j][8] for j in range(len(info_vecs))]
        course_z_score_local_end_time =[info_vecs[j][9] for j in range(len(info_vecs))]
        
        
        max_watching_counts, mean_watching_counts, std_watching_counts = max_mean_std(course_watching_counts)
        max_video_durations, mean_video_durations, std_video_durations = max_mean_std(course_video_durations)
        max_local_watching_times, mean_local_watching_times, std_local_watching_times = max_mean_std(course_local_watching_times)
        max_video_progress_times, mean_video_progress_times, std_video_progress_times = max_mean_std(course_video_progress_times)
        max_video_watching_speed, mean_video_watching_speed, std_video_watching_speed = max_mean_std(course_video_watching_speed)
        max_video_start_times, mean_video_start_times, std_video_start_times = max_mean_std(course_video_start_times)
        max_video_end_times, mean_video_end_times, std_video_end_times = max_mean_std(course_video_end_times)
        max_z_score_interval, mean_z_score_interval, std_z_score_interval = max_mean_std(course_z_score_interval)
        max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time = max_mean_std(course_z_score_local_start_time)
        max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time = max_mean_std(course_z_score_local_end_time)
        
        max_time_per_watching, mean_time_per_watching, std_time_per_watching = max_mean_std(course_time_per_watching)
        
        #compared to global stastic

        course_frequence_z_score = Z_score(mean_course_frequence,std_course_frequence,course_frequence[courses_[i]])
    
        total_watching_counts = np.sum(course_watching_counts)
        total_video_durations = np.sum(course_video_durations)
        total_local_watching_times = np.sum(course_local_watching_times)
        total_video_progress_times = np.sum(course_video_progress_times)
        
        vec = [courses_[i],num_of_cos,num_of_vdo,max_watching_counts, mean_watching_counts, std_watching_counts,
                max_video_durations, mean_video_durations, std_video_durations,
                max_local_watching_times, mean_local_watching_times, std_local_watching_times,
                max_video_progress_times, mean_video_progress_times, std_video_progress_times,
                max_video_watching_speed, mean_video_watching_speed, std_video_watching_speed, 
                max_video_start_times, mean_video_start_times, std_video_start_times,
                max_video_end_times, mean_video_end_times, std_video_end_times,
                max_z_score_interval, mean_z_score_interval, std_z_score_interval, 
                total_video_num, percentage_viewed,
                max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time,
                max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time,
#                 mean_watching_counts_ratio,mean_video_durations_ratio,mean_local_watching_times_ratio,mean_watching_speed_ratio,
#                 std_watching_counts_ratio,std_video_durations_ratio,std_local_watching_times_ratio,std_watching_speed_ratio
               course_frequence_z_score,max_time_per_watching, mean_time_per_watching, std_time_per_watching,
               total_watching_counts,total_video_durations,total_local_watching_times,total_video_progress_times
               
              ]
        
        course_vec.append(vec)
    

    r_lightGBM =  model_lgb.predict_proba(course_vec).tolist()
    r1 = []
    for i in range(len(r_lightGBM)):
        r1.append(r_lightGBM[i][1])
#     r = model_lgb.predict(course_vec).tolist()
#     r = catboost_model.predict(course_vec).tolist()
    
    return r1


train_set['course_vecs2'] = train_set.progress_apply(lambda row: feature_genration(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)



data = train_set[['label_list','course_vecs2']]
labels = data['label_list'].values.tolist()
y = [ item for elem in labels for item in elem]
course_info = data['course_vecs2'].values.tolist()
course_list = [ item for elem in course_info for item in elem]

model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)

model_lgb.fit(course_list, y, 
                  eval_names=['train'],
                  eval_metric=['logloss','auc'],
                  eval_set=[(course_list, y)],
                  early_stopping_rounds=10)


#test data prep
test_set['course_vecs2'] = test_set.progress_apply(lambda row: feature_genration(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)
test_data = train_set[['label_list','course_vecs2']]

test_labels = test_data['label_list'].values.tolist()
ground_truth = [ item for elem in test_labels for item in elem]

course_info_test = test_data['course_vecs2'].values.tolist()
course_list_test = [ item for elem in course_info_test for item in elem]

result2 =  model_lgb.predict(course_list_test)
result2 =result2.tolist()
acc = calculate_acc(result2,ground_truth)


nb_courses = 706+1
course_emb_size = 5
nb_videos = 38181+1
video_emb_size = 15
feature_size1 = course_emb_size + 42
sequence_len = 70
feature_size2 = course_emb_size + video_emb_size + 13
num_out_channel = 32
kernel_size = [3,4,5]
output_size = 32
hidden_dim = 64
num_of_lstm_layer = 1



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.course_embedding = torch.nn.Embedding(nb_courses, course_emb_size)
        self.video_embedding = torch.nn.Embedding(nb_videos, video_emb_size)
        
        self.ReLU_activation =  nn.ReLU()
        self.tanh_activation =  nn.Tanh()
        
        
        self.conv1 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[0])
        self.maxpool1 = nn.MaxPool1d(sequence_len - kernel_size[0] + 1) 
        self.conv2 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[1])
        self.maxpool2 = nn.MaxPool1d(sequence_len - kernel_size[1] + 1) 
        self.conv3 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[2])
        self.maxpool3 = nn.MaxPool1d(sequence_len - kernel_size[2] + 1)
        self.fc1 = nn.Linear(num_out_channel*len(kernel_size), 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
#         self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()


    def forward(self, course_id, video_id, continues):
        #course_id  (batch_size, max_sen_len)
        #continues  (batch_size, max_sen_len, feature_size)
        emb1 = self.course_embedding(course_id) # (batch_size,max_sen_len, embed_size)
        emb2 = self.video_embedding(video_id)

        x = torch.cat([emb1,emb2,continues], 2)
        x = x.permute(0, 2, 1)  # Batch_size * (feature_dim) * max_sen_len

        x1 = self.conv1(x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x1 = self.ReLU_activation(x1)
        x1 = self.maxpool1(x1)
        x1 = x1.squeeze(2) 
        
        x2 = self.conv2(x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x2 = self.ReLU_activation(x2)
        x2 = self.maxpool2(x2)
        x2 = x2.squeeze(2)
        
        x3 = self.conv3(x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x3 = self.ReLU_activation(x3)
        x3 = self.maxpool3(x3)
        x3 = x3.squeeze(2)

        all_out = torch.cat((x1, x2, x3), dim=1)
#         print(all_out.shape)
        info_fusion = self.tanh_activation(self.fc1(all_out))       
        info_fusion = self.tanh_activation(self.fc2(info_fusion)) 
        final_out = self.fc3(info_fusion)
#         result = self.softmax(final_out)
        result = self.sigmoid(final_out)

        return result  # 返回 softmax 的结果



model_CNN = TextCNN()
model_CNN.load_state_dict(torch.load('textcnn_25epoch.model'))
model_CNN.eval()




def prediction_train_seperately(course_vecs_CNN):
    course_id = []
    video_id = []
    continues_feature = []
     

        
        
    #get x for CNN
    course_list = course_vecs_CNN
    course_id_CNN = []
    video_id = []
    continues_feature2 = []

    for i in range(len(course_list)): #get a course 
        c = course_list[i]
        course_cat1 = []
        course_cat2 = []
        course_con = []
        for j in range(len(c)):       #get a subject
            s = c[j]
            cat_feture1 = s[0]       #get course_id and video_id
            cat_feture2 = s[1]
            course_cat1.append(cat_feture1)
            course_cat2.append(cat_feture2)
            con_feture = s[2:]        #get continues features
            course_con.append(con_feture)
        if len(course_cat1)<sequence_len:
            length = sequence_len - len(course_cat1)
            temp_course_id = [706] * length
            temp_video_id = [38180] * length
            temp2 = [[0,0,0,0,0,0,0,0,0,0,0,0,0]] * length
            course_cat1 = course_cat1 + temp_course_id
            course_cat2 = course_cat2 + temp_video_id
            course_con = course_con + temp2

        course_id_CNN.append(course_cat1) 
        video_id.append(course_cat2) 
        continues_feature2.append(course_con)

    # to tensor
    
    
    continues_feature2 = torch.tensor(continues_feature2)
    course_id_CNN = torch.tensor(course_id_CNN).clone().long()
    video_id = torch.tensor(video_id).clone().long()
    
    
    predictions_gru = model_BiGRU(course_id_CNN,video_id,continues_feature2)

    predictions = torch.flatten(predictions_gru)
   
    
    results_prob = predictions.detach().numpy().tolist()

    

    return results_prob


def merge_results(result1,result2):
    result = []
    for i in range(len(result1)):
        result.append( int(np.round((result1[i]+result2[i])/2)) )
    return result

def merge_results_prob(result1,result2):
    result = []
    for i in range(len(result1)):
        result.append( (result1[i]+result2[i])/2 )
    return result



test_set['gbdt_result'] = test_set.progress_apply(lambda row: gbdt_prediction(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)



test_set['textcnn_result'] = test_set['course_vecs_CNN'].progress_apply(lambda x : prediction_train_seperately(x))



test_set['predictions'] = test_set.progress_apply(lambda row: merge_results(row['textcnn_result'],row['gbdt_result']), axis=1)

test_set['predictions_prob'] = test_set.progress_apply(lambda row: merge_results_prob(row['textcnn_result'],row['gbdt_result']), axis=1)



final_result = test_set[['predictions']].values.tolist()
final_result_prob = test_set[['predictions_prob']].values.tolist()
ground_truth = test_set[['label_list']].values.tolist()
# test = final_result.values.tolist()
final_result = [ item for elem in final_result for item in elem]
final_result = [ item for elem in final_result for item in elem]
final_result_prob = [ item for elem in final_result_prob for item in elem]
final_result_prob = [ item for elem in final_result_prob for item in elem]
ground_truth = [ item for elem in ground_truth for item in elem]
ground_truth = [ item for elem in ground_truth for item in elem]
acc = calculate_acc(final_result,ground_truth)



auc = roc_auc_score(ground_truth, final_result_prob)



f1=f1_score(ground_truth, final_result, average='macro')






