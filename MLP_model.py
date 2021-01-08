#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pandas_profiling
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from joblib import Parallel,delayed
import numpy as np
import re
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn.utils import shuffle

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


train_set_course_vec_ = pd.read_csv('train_set_course_vec.csv',converters={'label_list': eval,'course_vecs_CNN': eval})
test_set_course_vec_ = pd.read_csv('test_set_course_vec.csv',converters={'label_list': eval,'course_vecs_CNN': eval})


train_set_course_vec = train_set_course_vec_[['label_list','course_vecs_CNN']]
del train_set_course_vec_

test_set_course_vec = test_set_course_vec_[['label_list','course_vecs_CNN']]
del test_set_course_vec_



def training_data_prep():
    course_id = []
    video_id = []
    continues_feature = []
    data = train_set_course_vec[['label_list','course_vecs_CNN']]

    data = shuffle(data) #Shuffle data
    #get y
    labels = data['label_list'].values.tolist()
    y = [ item for elem in labels for item in elem]
    #get x
    course_info = data['course_vecs_CNN'].values.tolist()
    course_list = [ item for elem in course_info for item in elem]
#     print(course_list[0][0])
    course_id = []
    video_id = []
    continues_feature = []
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

        course_id.append(course_cat1) 
        video_id.append(course_cat2) 
        continues_feature.append(course_con)

    # to tensor
    continues_feature = torch.tensor(continues_feature)
    course_id = torch.tensor(course_id)
    video_id = torch.tensor(video_id)
    y = torch.tensor(y)
    return continues_feature,course_id,video_id,y



def test_data_prep():
    course_id = []
    video_id = []
    continues_feature = []
    data = test_set_course_vec[['label_list','course_vecs_CNN']]

#     data = shuffle(data) #Shuffle data
    #get y
    labels = data['label_list'].values.tolist()
    y = [ item for elem in labels for item in elem]
    #get x
    course_info = data['course_vecs_CNN'].values.tolist()
    course_list = [ item for elem in course_info for item in elem]
    course_id = []
    video_id = []
    continues_feature = []
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
        if len(course_cat1) < sequence_len:
            length = sequence_len - len(course_cat1)
            temp_course_id = [706] * length
            temp_video_id = [38180] * length
            temp2 = [[0,0,0,0,0,0,0,0,0,0,0,0,0]] * length
            course_cat1 = course_cat1 + temp_course_id
            course_cat2 = course_cat2 + temp_video_id
            course_con = course_con + temp2

        course_id.append(course_cat1) 
        video_id.append(course_cat2) 
        continues_feature.append(course_con)


    # to tensor
    continues_feature = torch.tensor(continues_feature)
    course_id = torch.tensor(course_id)
    video_id = torch.tensor(video_id)
    y = torch.tensor(y)
    return continues_feature,course_id,video_id,y



nb_courses = 706+1
course_emb_size = 5
nb_videos = 38181+1
video_emb_size = 15
sequence_len = 70
# in_channel = 
feature_size2 = course_emb_size + video_emb_size + 13
hidden_dim = 32
num_of_lstm_layer = 1
# batchSize = 512



class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        
        
        self.course_embedding = torch.nn.Embedding(nb_courses, course_emb_size)
        self.video_embedding = torch.nn.Embedding(nb_videos, video_emb_size)
        
        self.ReLU_activation =  nn.ReLU()
        self.tanh_activation =  nn.Tanh()
        self.sigmoid_activation = nn.Sigmoid()
        
        
#         self.bi_gru = nn.GRU(input_size = feature_size, hidden_size = hidden_dim, num_layers=num_of_lstm_layer,  bidirectional=False)
        
        
        self.fc1 = nn.Linear(feature_size2*sequence_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, course_id, video_id, continues,b_size):
        
        #course_id  (batch_size, max_sen_len)
        #continues  (batch_size, max_sen_len, feature_size)
        emb1 = self.course_embedding(course_id) # (batch_size,max_sen_len, embed_size)
        emb2 = self.video_embedding(video_id)
        

        x = torch.cat([emb1,emb2,continues], 2)
        
        input_x = x.view(b_size,-1)
        
        info_fusion = self.tanh_activation(self.fc1(input_x))
        
        
        info_fusion = self.tanh_activation(self.fc2(info_fusion))
        
        final_out = self.fc3(info_fusion)

        result = self.sigmoid(final_out)

        return result



model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# MSELoss = nn.MSELoss()
MSELoss = nn.BCELoss()

epoach_count = 25 #40
batchSize = 512
loss_value = []
acc_value = []


test_continues_feature, test_course_id, test_video_id, ground_truth = test_data_prep()
ground_truth = ground_truth.detach().numpy().tolist()


for epoach in range(epoach_count):
    continues_feature,course_id,video_id,y = training_data_prep()
    numOfMinibatches = int(len(course_id) / batchSize) + 1
    numOfLastMinibatch = len(course_id) % batchSize
#     loss_value = []
    for batchID in range(numOfMinibatches):
        if batchID == numOfMinibatches-1:
            numbOfBatches = numOfLastMinibatch
        else:
            numbOfBatches = batchSize
        leftIndex = batchID * batchSize
        rightIndex = leftIndex + numbOfBatches
        courseid =  course_id[leftIndex: rightIndex].clone().long()
        videoid =  video_id[leftIndex: rightIndex].clone().long()
        continuesfeature =  continues_feature[leftIndex: rightIndex].clone()
        
        predictions = model(courseid,videoid,continuesfeature,numbOfBatches)
#         predictions = torch.round(torch.flatten(predictions))
        predictions = torch.flatten(predictions)
#         print('prediction: ',predictions)
#         print('y: ',y[leftIndex: rightIndex])
#         loss = BCrossEntropyLoss(predictions,y[leftIndex: rightIndex].float())
        loss = MSELoss(predictions,y[leftIndex: rightIndex].float())
#         print('loss: ',loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
        
        
        #testing
        
        if(batchID%100==0):
            test_numOfMinibatches = int(len(test_course_id) / batchSize) + 1
            test_numOfLastMinibatch = len(test_course_id) % batchSize
            results = []
            for test_batchID in range(test_numOfMinibatches):
                if test_batchID == test_numOfMinibatches-1:
                    test_numbOfBatches = test_numOfLastMinibatch
                else:
                    test_numbOfBatches = batchSize
                test_leftIndex = test_batchID * batchSize
                test_rightIndex = test_leftIndex + test_numbOfBatches
                test_courseid =  test_course_id[test_leftIndex: test_rightIndex].clone().long()
                test_videoid =  test_video_id[test_leftIndex: test_rightIndex].clone().long()
                test_continuesfeature =  test_continues_feature[test_leftIndex: test_rightIndex].clone()
#                 print('test_numOfMinibatches: ',test_numOfMinibatches)
#                 print('test_numOfLastMinibatch: ',test_numOfLastMinibatch)
#                 print('test_rightIndex: ',test_rightIndex)
#                 print('test_leftIndex: ',test_leftIndex)
                test_predictions = model(test_courseid,test_videoid,test_continuesfeature,test_numbOfBatches)
                test_predictions = torch.round(torch.flatten(test_predictions))
                results.append(test_predictions.detach().numpy().tolist())
            result = [ item for elem in results for item in elem]
#             ground_truth = ground_truth.detach().numpy().tolist()
            acc = calculate_acc(result,ground_truth)
            acc_value.append(acc)
            print('Epoch[{}/{}],loss:{:.4f},acc:{:.4f}'.format(epoach, epoach_count,loss.item(),acc))

#         batchIndex = batchList[leftIndex: rightIndex]


torch.save(model.state_dict(), 'mlp.model')




