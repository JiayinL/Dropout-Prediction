#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import pandas_profiling
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from joblib import Parallel,delayed
import numpy as np
import json
import re
import time
from sklearn.utils import shuffle
# tqdm.pandas()



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
test_set = pd.read_csv('test_set.csv',converters={'label_list': eval})
course_info = pd.read_json('course_info.json',lines=True)  
video_info = pd.read_json('video_info.json',lines=True)  

videoID = video_info['id'].values.tolist()
courseID = course_info['course_id'].values.tolist()
videoID_encoder = LabelEncoder()
courseID_encoder = LabelEncoder()
videoID_encoder.fit(videoID)
courseID_encoder.fit(courseID)
course_info['courseID'] = course_info['course_id'].progress_apply(lambda x : courseID_encoder.transform([x]))
course_info['videoIDs'] = course_info['item'].progress_apply(lambda x : videoID_encoder.transform(x))
video_info['videoID'] = video_info['id'].progress_apply(lambda x : videoID_encoder.transform([x]))


course_video_num = {}
def count_videos(courseId, videoIds):
    number_of_video = len(videoIds)
    course_video_num[courseId[0]] = number_of_video
    
course_info.progress_apply(lambda row: count_videos(row['courseID'],row['videoIDs']), axis=1)   

video_lists = {}
def init_video_list(videoID):
    video_lists[videoID[0]] = []
    
# video_info.progress_apply(lambda row: init_video_list(row['courseID'],row['videoIDs']), axis=1)  
video_info['videoID'].progress_apply(lambda x : init_video_list(x))

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

#2020-10-24 new feature 
video_watching_speed_info = []
per_time_watching_percentage_info = [] #平均每次观看视频完成百分比
video_finish_percentage_info = [] #完成观看视频百分比



num_of_cos_info = []
num_of_vdo_info = []
local_max_watching_counts_info = []
local_mean_watching_counts_info = []
local_std_watching_counts_info = []
local_max_video_durations_info = []
local_mean_video_durations_info = []
local_std_video_durations_info = []
local_max_local_watching_times_info = []
local_mean_local_watching_times_info = []
local_std_local_watching_times_info = []
local_max_video_progress_times_info = []
local_mean_video_progress_times_info = []
local_std_video_progress_times_info = []
local_max_video_watching_speed_info = []
local_mean_video_watching_speed_info = [] 
local_std_video_watching_speed_info = [] 
local_max_video_start_times_info = [] 
local_mean_video_start_times_info = [] 
local_std_video_start_times_info = []
local_max_video_end_times_info = []
local_mean_video_end_times_info = [] 
local_std_video_end_times_info = []
local_max_z_score_interval_info = []
local_mean_z_score_interval_info = [] 
local_std_z_score_interval_info = []
local_total_video_num_info = []
local_percentage_viewed_info = []
local_max_z_score_local_start_time_info = []
local_mean_z_score_local_start_time_info = [] 
local_std_z_score_local_start_time_info = []
local_max_z_score_local_end_time_info = [] 
local_mean_z_score_local_end_time_info = []
local_std_z_score_local_end_time_info = []
local_course_frequence_z_score_info = []
local_max_time_per_watching_info = [] 
local_mean_time_per_watching_info = [] 
local_std_time_per_watching_info = []
local_total_watching_counts_info = []
local_total_video_durations_info = []
local_total_local_watching_times_info = []
local_total_video_progress_times_info = []

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
        local_watching_times_info.append(local_watching_times[i])
        video_progress_times_info.append(video_progress_times[i])
        watching_counts_info.append(watching_counts[i])
        local_interval_info.append(interval)
            
            #2020-10-24 new feature 
        video_watching_speed_info.append(local_watching_times[i]/video_progress_times[i])
        per_time_watching_percentage_info.append((local_watching_times[i]/watching_counts[i])/video_durations[i])
        video_finish_percentage_info.append(local_watching_times[i]/video_durations[i])
            #2020-10-26 new feature 
        video_lists[video_ids[i]].append(local_watching_times[i])


        info_vec = [course_ids[i],watching_counts[i],video_durations[i],local_watching_times[i],video_progress_times[i],
                    video_start_times[i], video_end_times[i]]   
    
        courses[course].append(info_vec)    

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
#         course_z_score_interval = [info_vecs[j][7] for j in range(len(info_vecs))]
#         course_z_score_local_start_time =[info_vecs[j][8] for j in range(len(info_vecs))]
#         course_z_score_local_end_time =[info_vecs[j][9] for j in range(len(info_vecs))]
        
        
        max_watching_counts, mean_watching_counts, std_watching_counts = max_mean_std(course_watching_counts)    
        max_video_durations, mean_video_durations, std_video_durations = max_mean_std(course_video_durations)
        max_local_watching_times, mean_local_watching_times, std_local_watching_times = max_mean_std(course_local_watching_times)
        max_video_progress_times, mean_video_progress_times, std_video_progress_times = max_mean_std(course_video_progress_times)
        max_video_watching_speed, mean_video_watching_speed, std_video_watching_speed = max_mean_std(course_video_watching_speed)
        max_video_start_times, mean_video_start_times, std_video_start_times = max_mean_std(course_video_start_times)
        max_video_end_times, mean_video_end_times, std_video_end_times = max_mean_std(course_video_end_times)
        
#         max_z_score_interval, mean_z_score_interval, std_z_score_interval = max_mean_std(course_z_score_interval)
#         max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time = max_mean_std(course_z_score_local_start_time)
#         max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time = max_mean_std(course_z_score_local_end_time)
        
        max_time_per_watching, mean_time_per_watching, std_time_per_watching = max_mean_std(course_time_per_watching)

#         course_frequence_z_score = Z_score(mean_course_frequence,std_course_frequence,course_frequence[courses_[i]])
        total_watching_counts = np.sum(course_watching_counts)
        total_video_durations = np.sum(course_video_durations)
        total_local_watching_times = np.sum(course_local_watching_times)
        total_video_progress_times = np.sum(course_video_progress_times)
        
        num_of_cos_info.append(num_of_cos)
        num_of_vdo_info.append(num_of_vdo)
        local_max_watching_counts_info.append(max_watching_counts)
        local_mean_watching_counts_info.append(mean_watching_counts)
        local_std_watching_counts_info.append(std_watching_counts)
        local_max_video_durations_info.append(max_video_durations)
        local_mean_video_durations_info.append(mean_video_durations)
        local_std_video_durations_info.append(std_video_durations)
        local_max_local_watching_times_info.append(max_local_watching_times)
        local_mean_local_watching_times_info.append(mean_local_watching_times)
        local_std_local_watching_times_info.append(std_local_watching_times)
        local_max_video_progress_times_info.append(max_video_progress_times)
        local_mean_video_progress_times_info.append(mean_video_progress_times)
        local_std_video_progress_times_info.append(std_video_progress_times)
        local_max_video_watching_speed_info.append(max_video_watching_speed)
        local_mean_video_watching_speed_info.append(mean_video_watching_speed) 
        local_std_video_watching_speed_info.append(std_video_watching_speed)
        local_max_video_start_times_info.append(max_video_start_times) 
        local_mean_video_start_times_info.append(mean_video_start_times) 
        local_std_video_start_times_info.append(std_video_start_times)
        local_max_video_end_times_info.append(max_video_end_times)
        local_mean_video_end_times_info.append(mean_video_end_times) 
        local_std_video_end_times_info.append(std_video_end_times)
#         max_z_score_interval_info = []
#         mean_z_score_interval_info = [] 
#         std_z_score_interval_info = []
        local_total_video_num_info.append(total_video_num)
        local_percentage_viewed_info.append(percentage_viewed)
#         max_z_score_local_start_time_info = []
#         mean_z_score_local_start_time_info = [] 
#         std_z_score_local_start_time_info = []
#         max_z_score_local_end_time_info = [] 
#         mean_z_score_local_end_time_info = []
#         std_z_score_local_end_time_info = []
#         course_frequence_z_score_info = []
        local_max_time_per_watching_info.append(max_time_per_watching) 
        local_mean_time_per_watching_info.append(mean_time_per_watching)
        local_std_time_per_watching_info.append(std_time_per_watching)
        local_total_watching_counts_info.append(total_watching_counts)
        local_total_video_durations_info.append(total_video_durations)
        local_total_local_watching_times_info.append(total_local_watching_times)
        local_total_video_progress_times_info.append(total_video_progress_times)
            


train_set.progress_apply(lambda row: collect_info(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)            

    #global stastic info for CNN
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

#2020-10-24 new feature
std_video_watching_speed_info = np.std(video_watching_speed_info, ddof=1)
mean_video_watching_speed_info = np.mean(video_watching_speed_info)

std_per_time_watching_percentage_info = np.std(per_time_watching_percentage_info, ddof=1)
mean_per_time_watching_percentage_info = np.mean(per_time_watching_percentage_info)

std_video_finish_percentage_info = np.std(video_finish_percentage_info, ddof=1)
mean_video_finish_percentage_info = np.mean(video_finish_percentage_info)


#2020-10-26 new feature 
std_local_watching_times_info2 = {}  #for a specific video locally
mean_local_watching_times_info2 = {}
for k,v in video_lists.items():
    vid = k
    watching_time_list = v
    if len(watching_time_list)> 1:
        mean = np.mean(watching_time_list)
        std = np.std(watching_time_list, ddof=1)
        std_local_watching_times_info2[vid] = std
        mean_local_watching_times_info2[vid] = mean
#     elif len(watching_time_list) = 1:
#         std_local_watching_times_info2[vid] = -1
#         mean_local_watching_times_info2[vid] = watching_time_list[0]
    else:
        std_local_watching_times_info2[vid] = -1
        mean_local_watching_times_info2[vid] = -1



#global stastic info for LR
std_num_of_cos_info = np.std(num_of_cos_info, ddof =1)
mean_num_of_cos_info = np.mean(num_of_cos_info)

std_num_of_vdo_info = np.std(num_of_vdo_info, ddof=1)
mean_num_of_vdo_info = np.mean(num_of_vdo_info)

std_max_watching_counts_info = np.std(local_max_watching_counts_info, ddof = 1)
mean_max_watching_counts_info = np.std(local_max_watching_counts_info)

std_mean_watching_counts_info = np.std(local_mean_watching_counts_info, ddof =1)
mean_mean_watching_counts_info = np.mean(local_mean_watching_counts_info)

std_std_watching_counts_info = np.std(local_std_watching_counts_info, ddof =1)
mean_std_watching_counts_info = np.mean(local_std_watching_counts_info)

std_max_video_durations_info = np.std(local_max_video_durations_info, ddof =1)
mean_max_video_durations_info = np.mean(local_max_video_durations_info)

std_mean_video_durations_info = np.std(local_mean_video_durations_info, ddof=1)
mean_mean_video_durations_info = np.mean(local_mean_video_durations_info)

std_std_video_durations_info = np.std(local_std_video_durations_info, ddof = 1)
mean_std_video_durations_info = np.mean(local_std_video_durations_info)

std_max_local_watching_times_info = np.std(local_max_local_watching_times_info, ddof =1)
mean_max_local_watching_times_info = np.mean(local_max_local_watching_times_info)

std_mean_local_watching_times_info = np.std(local_mean_local_watching_times_info,ddof=1)
mean_mean_local_watching_times_info = np.mean(local_mean_local_watching_times_info)

std_std_local_watching_times_info = np.std(local_std_local_watching_times_info,ddof = 1)
mean_std_local_watching_times_info = np.mean(local_std_local_watching_times_info)

std_max_video_progress_times_info = np.std(local_max_video_progress_times_info, ddof=1)
mean_max_video_progress_times_info = np.mean(local_max_video_progress_times_info)

std_mean_video_progress_times_info = np.std(local_mean_video_progress_times_info, ddof=1)
mean_mean_video_progress_times_info = np.mean(local_mean_video_progress_times_info)

std_std_video_progress_times_info = np.std(local_std_video_progress_times_info,ddof=1)
mean_std_video_progress_times_info = np.mean(local_std_video_progress_times_info)

std_max_video_watching_speed_info = np.std(local_max_video_watching_speed_info, ddof=1)
mean_max_video_watching_speed_info = np.mean(local_max_video_watching_speed_info)

std_mean_video_watching_speed_info = np.std(local_mean_video_watching_speed_info,ddof=1)
mean_mean_video_watching_speed_info = np.mean(local_mean_video_watching_speed_info)

std_std_video_watching_speed_info = np.std(local_std_video_watching_speed_info,ddof=1)
mean_std_video_watching_speed_info = np.mean(local_std_video_watching_speed_info)

std_max_video_start_times_info = np.std(local_max_video_start_times_info,ddof=1)
mean_max_video_start_times_info = np.mean(local_max_video_start_times_info)

std_mean_video_start_times_info = np.std(local_mean_video_start_times_info, ddof=1)
mean_mean_video_start_times_info = np.mean(local_mean_video_start_times_info)

std_std_video_start_times_info = np.std(local_std_video_start_times_info, ddof=1)
mean_std_video_start_times_info = np.mean(local_std_video_start_times_info)

std_max_video_end_times_info = np.std(local_max_video_end_times_info, ddof=1)
mean_max_video_end_times_info = np.mean(local_max_video_end_times_info)

std_mean_video_end_times_info = np.std(local_mean_video_end_times_info, ddof=1)
mean_mean_video_end_times_info = np.mean(local_mean_video_end_times_info)

std_std_video_end_times_info = np.std(local_std_video_end_times_info, ddof=1)
mean_std_video_end_times_info = np.mean(local_std_video_end_times_info)

std_total_video_num_info = np.std(local_total_video_num_info, ddof=1)
mean_total_video_num_info = np.mean(local_total_video_num_info)

std_percentage_viewed_info = np.std(local_percentage_viewed_info, ddof=1)
mean_percentage_viewed_info = np.mean(local_percentage_viewed_info)

std_max_time_per_watching_info = np.std(local_max_time_per_watching_info, ddof=1)
mean_max_time_per_watching_info = np.mean(local_max_time_per_watching_info)

std_mean_time_per_watching_info = np.std(local_mean_time_per_watching_info, ddof=1)
mean_mean_time_per_watching_info = np.mean(local_mean_time_per_watching_info)

std_std_time_per_watching_info = np.std(local_std_time_per_watching_info, ddof =1)
mean_std_time_per_watching_info = np.mean(local_std_time_per_watching_info)

std_total_watching_counts_info = np.std(local_total_watching_counts_info, ddof =1)
mean_total_watching_counts_info = np.mean(local_total_watching_counts_info)

std_total_video_durations_info = np.std(local_total_video_durations_info, ddof=1)
mean_total_video_durations_info = np.mean(local_total_video_durations_info)

std_total_local_watching_times_info = np.std(local_total_local_watching_times_info, ddof=1)
mean_total_local_watching_times_info = np.mean(local_total_local_watching_times_info)

std_total_video_progress_times_info = np.std(local_total_video_progress_times_info, ddof=1)
mean_total_video_progress_times_info = np.mean(local_total_video_progress_times_info)



def feature_genration_LR(course_ids,video_ids,watching_counts,video_durations,local_watching_times,video_progress_times,
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
        
        vec = [courses_[i],Z_score(mean_num_of_cos_info,std_num_of_cos_info,num_of_cos),
                Z_score(mean_num_of_vdo_info,std_num_of_vdo_info,num_of_vdo),
                Z_score(mean_max_watching_counts_info,std_max_watching_counts_info,max_watching_counts), 
                Z_score(mean_mean_watching_counts_info,std_mean_watching_counts_info,mean_watching_counts), 
                Z_score(mean_std_watching_counts_info,std_std_watching_counts_info,std_watching_counts),
                Z_score(mean_max_video_durations_info,std_max_video_durations_info,max_video_durations), 
                Z_score(mean_mean_video_durations_info,std_mean_video_durations_info,mean_video_durations), 
                Z_score(mean_std_video_durations_info,std_std_video_durations_info,std_video_durations),
                Z_score(mean_max_local_watching_times_info,std_max_local_watching_times_info,max_local_watching_times), 
                Z_score(mean_mean_local_watching_times_info,std_mean_local_watching_times_info,mean_local_watching_times), 
                Z_score(mean_std_local_watching_times_info,std_std_local_watching_times_info,std_local_watching_times),
                Z_score(mean_max_video_progress_times_info,std_max_video_progress_times_info,max_video_progress_times), 
                Z_score(mean_mean_video_progress_times_info,std_mean_video_progress_times_info,mean_video_progress_times), 
                Z_score(mean_std_video_progress_times_info,std_std_video_progress_times_info,std_video_progress_times),
                Z_score(mean_max_video_watching_speed_info,std_max_video_watching_speed_info,max_video_watching_speed), 
                Z_score(mean_mean_video_watching_speed_info,std_mean_video_watching_speed_info,mean_video_watching_speed), 
                Z_score(mean_std_video_watching_speed_info,std_std_video_watching_speed_info,std_video_watching_speed), 
                Z_score(mean_max_video_start_times_info,std_max_video_start_times_info,max_video_start_times),
                Z_score(mean_mean_video_start_times_info,std_mean_video_start_times_info,mean_video_start_times), 
                Z_score(mean_std_video_start_times_info,std_std_video_start_times_info,std_video_start_times),
                Z_score(mean_max_video_end_times_info,std_max_video_end_times_info,max_video_end_times), 
                Z_score(mean_mean_video_end_times_info,std_mean_video_end_times_info,mean_video_end_times), 
                Z_score(mean_std_video_end_times_info,std_std_video_end_times_info,std_video_end_times),
                max_z_score_interval, mean_z_score_interval, std_z_score_interval, 
                Z_score(mean_total_video_num_info,std_total_video_num_info,total_video_num),
                Z_score(mean_percentage_viewed_info,std_percentage_viewed_info,percentage_viewed),
                max_z_score_local_start_time, mean_z_score_local_start_time, std_z_score_local_start_time,
                max_z_score_local_end_time, mean_z_score_local_end_time, std_z_score_local_end_time,
                course_frequence_z_score,
                Z_score(mean_max_time_per_watching_info,std_max_time_per_watching_info,max_time_per_watching), 
                Z_score(mean_mean_time_per_watching_info,std_mean_time_per_watching_info,mean_time_per_watching), 
                Z_score(mean_std_time_per_watching_info,std_std_time_per_watching_info,std_time_per_watching),
                Z_score(mean_total_watching_counts_info,std_total_watching_counts_info,total_watching_counts),
                Z_score(mean_total_video_durations_info,std_total_video_durations_info,total_video_durations),
                Z_score(mean_total_local_watching_times_info,std_total_local_watching_times_info,total_local_watching_times),
                Z_score(mean_total_video_progress_times_info,std_total_video_progress_times_info,total_video_progress_times)
               ]
        
        course_vec.append(vec)

    
    return course_vec



def feature_generate_CNN(course_ids,video_ids,watching_counts,video_durations,local_watching_times,video_progress_times,
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
#         courses_textual_name[courses_[i]] = []
#         courses_textual_content[courses_[i]] = []
    for i in range(len(course_ids)):
        course = course_ids[i]
        if len(courses[course])< 70:
            local_start_times_ = time_transform(local_start_times[i])
            local_end_times_ = time_transform(local_end_times[i])
            interval = local_end_times_ - local_start_times_
            
            video_watching_speed = local_watching_times[i]/video_progress_times[i]
            per_time_watching_percentage = (local_watching_times[i]/watching_counts[i])/video_durations[i]
            video_finish_percentage = local_watching_times[i]/video_durations[i]
            
            std_local_watching_times2 = std_local_watching_times_info2[video_ids[i]]
            mean_local_watching_times2 = mean_local_watching_times_info2[video_ids[i]]
            if std_local_watching_times2 <= 0 or mean_local_watching_times2 <= 0: # cold start
                z_score_local_watching_times2 = 0
#             elif std_local_watching_times2 < 0 and mean_local_watching_times2 >=0: #only one historical record
#                 z_score_local_watching_times2 = local_watching_times[i]/(mean_local_watching_times2+1)
                
            else:
                z_score_local_watching_times2 = Z_score(mean_local_watching_times2,std_local_watching_times2,local_watching_times[i])


            video_vec = [course_ids[i],video_ids[i],
                        Z_score(mean_watching_counts_info,std_watching_counts_info,watching_counts[i]),
                        Z_score(mean_video_durations_info,std_video_durations_info,video_durations[i]),
                        Z_score(mean_local_watching_times_info,std_local_watching_times_info,local_watching_times[i]),
                        Z_score(mean_video_progress_times_info,std_video_progress_times_info,video_progress_times[i]),
                        Z_score(mean_video_start_times_info,std_video_start_times_info,video_start_times[i]),
                        Z_score(mean_video_end_times_info,std_video_end_times_info,video_end_times[i]), 
                        Z_score(mean_local_start_times_info,std_local_start_times_info,local_start_times_),
                        Z_score(mean_local_end_times_info,std_local_end_times_info,local_end_times_),
                        Z_score(mean_local_interval_info,std_local_interval_info,interval),
                        Z_score(mean_video_watching_speed_info,std_video_watching_speed_info,video_watching_speed),
                        Z_score(mean_per_time_watching_percentage_info,std_per_time_watching_percentage_info,per_time_watching_percentage),
                        Z_score(mean_video_finish_percentage_info,std_video_finish_percentage_info,video_finish_percentage),
                        z_score_local_watching_times2]
    
    
    
    
            courses[course].append(video_vec)

    course_vec = []

#     for i in range(len(courses_)):
#         course_vec.append(courses[courses_[i]])
        
    for i in range(len(courses_)):
        videos = courses[courses_[i]]
        temp = pd.DataFrame(videos, columns=['courseid','videoid','1','2','3','4','5','6','local_start_time','7','8','9','10','11','12']) 
        temp=temp.sort_values(by=['local_start_time'],ignore_index=True)
        c = temp.values.tolist()
        course_vec.append(c)

        
    return course_vec#,content_vec,name_vec



train_set['course_vecs_LR'] = train_set.progress_apply(lambda row: feature_genration_LR(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)
train_set['course_vecs_CNN'] = train_set.progress_apply(lambda row: feature_generate_CNN(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)
train_set.to_csv("train_set_course_vec.csv",index=0)


test_set['course_vecs_LR'] = test_set.progress_apply(lambda row: feature_genration_LR(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)

test_set['course_vecs_CNN'] = test_set.progress_apply(lambda row: feature_generate_CNN(row['course_ids'],row['video_ids'],row['watching_counts'],row['video_durations'],row['local_watching_times'],row['video_progress_times'],
                    row['video_start_times'], row['video_end_times'], row['local_start_times'],row['local_end_times'],row['courseListIDs']), axis=1)
test_set.to_csv("test_set_course_vec.csv",index=0)






