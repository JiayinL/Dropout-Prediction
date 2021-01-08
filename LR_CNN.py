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
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
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



train_set_course_vec = pd.read_csv('train_set_course_vec.csv',converters={'label_list': eval, 'course_vecs_CNN':eval, 'course_vecs_LR':eval})
test_set_course_vec = pd.read_csv('test_set_course_vec.csv',converters={'label_list': eval, 'course_vecs_CNN':eval, 'course_vecs_LR':eval})








nb_courses = 706+1
course_emb_size = 5
nb_videos = 38181+1
video_emb_size = 15


feature_size1 = course_emb_size + 42



sequence_len = 70
# in_channel = 
feature_size2 = course_emb_size + video_emb_size + 13
num_out_channel = 32
kernel_size = [3,4,5]
output_size = 32



def training_data_prep():
    course_id = []
    video_id = []
    continues_feature1 = []
    data = train_set_course_vec[['label_list','course_vecs_LR','course_vecs_CNN']]

    data = shuffle(data) #Shuffle data
    #get y
    labels = data['label_list'].values.tolist()
    y = [ item for elem in labels for item in elem]
    
    
    
    #get x for LR
    course_info_LR = data['course_vecs_LR'].values.tolist()

    course_id_LR = []
    continues_feature1 = []
    for i in range(len(course_info_LR)): #get a course
        c = course_info_LR[i]
        course_cat1 = []
        course_con = []
        for j in range(len(c)):       #get a subject
            s = c[j]
            cat_feture1 = s[0]       #get course_id and video_id
            course_cat1.append(cat_feture1)
            con_feture = s[1:]        #get continues features
            course_con.append(con_feture)
       
        course_id_LR.append(course_cat1) 
        continues_feature1.append(course_con)
        
        
    #get x for CNN
    course_info_CNN = data['course_vecs_CNN'].values.tolist()
    course_list = [ item for elem in course_info_CNN for item in elem]
#     print(course_list[0][0])
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
    continues_feature1 = [ item for elem in continues_feature1 for item in elem]
    course_id_LR = [ item for elem in course_id_LR for item in elem]
    
    continues_feature1 = torch.tensor(continues_feature1)
    course_id_LR = torch.tensor(course_id_LR)
    
    
    continues_feature2 = torch.tensor(continues_feature2)
    course_id_CNN = torch.tensor(course_id_CNN)
    video_id = torch.tensor(video_id)

    y = torch.tensor(y)
    return continues_feature1,continues_feature2,course_id_LR,course_id_CNN,video_id,y



def test_data_prep():
    course_id = []
    video_id = []
    continues_feature = []
    data = test_set_course_vec[['label_list','course_vecs_LR','course_vecs_CNN']]

    labels = data['label_list'].values.tolist()
    y = [ item for elem in labels for item in elem]
     
    
    #get x for LR
    course_info_LR = data['course_vecs_LR'].values.tolist()

    course_id_LR = []
    continues_feature1 = []
    for i in range(len(course_info_LR)): #get a course
        c = course_info_LR[i]
        course_cat1 = []
        course_con = []
        for j in range(len(c)):       #get a subject
            s = c[j]
            cat_feture1 = s[0]       #get course_id and video_id
            course_cat1.append(cat_feture1)
            con_feture = s[1:]        #get continues features
            course_con.append(con_feture)
       
        course_id_LR.append(course_cat1) 
        continues_feature1.append(course_con)
        
        
    #get x for CNN
    course_info_CNN = data['course_vecs_CNN'].values.tolist()
    course_list = [ item for elem in course_info_CNN for item in elem]
#     print(course_list[0][0])
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
    continues_feature1 = [ item for elem in continues_feature1 for item in elem]
    course_id_LR = [ item for elem in course_id_LR for item in elem]
    
    continues_feature1 = torch.tensor(continues_feature1)
    course_id_LR = torch.tensor(course_id_LR)
    
    
    continues_feature2 = torch.tensor(continues_feature2)
    course_id_CNN = torch.tensor(course_id_CNN)
    video_id = torch.tensor(video_id)

    y = torch.tensor(y)
    return continues_feature1,continues_feature2,course_id_LR,course_id_CNN,video_id,y



def prediction(course_vecs_LR,course_vecs_CNN):
    course_id = []
    video_id = []
    continues_feature = []


#     labels = data['label_list'].values.tolist()
#     y = [ item for elem in labels for item in elem]
     
    
    #get x for LR
    course_info_LR = course_vecs_LR

    course_id_LR = []
    continues_feature1 = []
    for i in range(len(course_info_LR)): #get a course
        c = course_info_LR[i]
    
        cat_feture1 = c[0]       #get course_id and video_id

        con_feture = c[1:]        #get continues features

       
        course_id_LR.append(cat_feture1) 
        continues_feature1.append(con_feture)
        
        
    #get x for CNN
#     course_info_CNN = course_vecs_CNN
    course_list = course_vecs_CNN
#     print(course_list[0][0])
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
    continues_feature1 = [ item for elem in continues_feature1 for item in elem]
    course_id_LR = [ item for elem in course_id_LR for item in elem]
    
    continues_feature1 = torch.tensor(continues_feature1)
    course_id_LR = torch.tensor(course_id_LR)
    
    
    continues_feature2 = torch.tensor(continues_feature2)
    course_id_CNN = torch.tensor(course_id_CNN)
    video_id = torch.tensor(video_id)

    y = torch.tensor(y)
    return continues_feature1,continues_feature2,course_id_LR,course_id_CNN,video_id,y



class LR_CNN(nn.Module):
    
    def __init__(self):
        super(LR_CNN, self).__init__()  
        
        self.course_embedding = torch.nn.Embedding(nb_courses, course_emb_size)
        self.video_embedding = torch.nn.Embedding(nb_videos, video_emb_size)
        
                
        self.conv1 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[0])
        self.maxpool1 = nn.MaxPool1d(sequence_len - kernel_size[0] + 1)
        
        
        self.conv2 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[1])
        self.maxpool2 = nn.MaxPool1d(sequence_len - kernel_size[1] + 1)
        
        self.conv3 = nn.Conv1d(in_channels=feature_size2,out_channels=num_out_channel,kernel_size=kernel_size[2])
        self.maxpool3 = nn.MaxPool1d(sequence_len - kernel_size[2] + 1)
        
#         self.conv4 = nn.Conv1d(in_channels=feature_size,out_channels=num_out_channel,kernel_size=kernel_size[3])
#         self.maxpool4 = nn.MaxPool1d(sequence_len - kernel_size[3] + 1)
        
        self.lr_fc = nn.Linear(feature_size1, 1)
        
        self.fc1 = nn.Linear(num_out_channel*len(kernel_size), 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.ReLU_activation =  nn.ReLU()
        self.tanh_activation =  nn.Tanh()
        self.sigmoid_activation = nn.Sigmoid()   
        
        self.final_fc = nn.Linear(2, 1)

    def forward(self, courseid_LR,courseid_CNN,continuesfeature1,continuesfeature2,videoid):
        
        #course_id  (batch_size, max_sen_len)
        #continues  (batch_size, max_sen_len, feature_size)
        emb1_LR = self.course_embedding(courseid_LR)
        emb1_CNN = self.course_embedding(courseid_CNN)# (batch_size,max_sen_len, embed_size)
        emb2 = self.video_embedding(videoid)
        
        # LR part
        LR_x = torch.cat([emb1_LR,continuesfeature1], 1)
        LR_result = self.sigmoid_activation(self.lr_fc(LR_x))
        
        #CNN part
#         print('emb1_CNN:',emb1_CNN.shape)
#         print('emb2:',emb2.shape)
#         print('continuesfeature2:',continuesfeature2.shape)
        
        CNN_x = torch.cat([emb1_CNN,emb2,continuesfeature2], 2)
        CNN_x = CNN_x.permute(0, 2, 1)  # Batch_size * (feature_dim) * max_sen_len
        x1 = self.conv1(CNN_x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x1 = self.ReLU_activation(x1)
        x1 = self.maxpool1(x1)
        x1 = x1.squeeze(2)
#         print('final x1: ',x1.shape)
       
        x2 = self.conv2(CNN_x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x2 = self.ReLU_activation(x2)
        x2 = self.maxpool2(x2)
        x2 = x2.squeeze(2)
        
        x3 = self.conv3(CNN_x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
        x3 = self.ReLU_activation(x3)
        x3 = self.maxpool3(x3)
        x3 = x3.squeeze(2)

#         x4 = self.conv4(CNN_x)#.squeeze(2)  # shape = (64, num_channels, 1)(squeeze 2)
#         x4 = self.ReLU_activation(x4)
#         x4 = self.maxpool4(x4)
#         x4 = x4.squeeze(2)        
        
        all_out = torch.cat((x1, x2, x3), dim=1)
        info_fusion = self.tanh_activation(self.fc1(all_out))
        info_fusion = self.tanh_activation(self.fc2(info_fusion)) 
        final_out = self.fc3(info_fusion)
        CNN_result = self.sigmoid_activation(final_out)
        
        
        #combine two result
        final_input = torch.cat((LR_result, CNN_result), dim=1)
#         print(final_input.shape)
        result = self.sigmoid_activation(self.final_fc(final_input))        
        
        return result,LR_result,CNN_result



test_continues_feature1,test_continues_feature2,test_course_id_LR,test_course_id_CNN,test_video_id,ground_truth = test_data_prep()
ground_truth = ground_truth.detach().numpy().tolist()




model = LR_CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# MSELoss = nn.MSELoss()
MSELoss = nn.BCELoss()

epoach_count = 15 #40
batchSize = 512
loss_value = []
acc_value = []
times = []



for m in model.modules():
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)




epoach_count = 5 #40



for epoach in range(epoach_count):
    start = time.time()
    continues_feature1,continues_feature2,course_id_LR,course_id_CNN,video_id,y = training_data_prep()
    numOfMinibatches = int(len(course_id_CNN) / batchSize) + 1
    numOfLastMinibatch = len(course_id_CNN) % batchSize
#     loss_value = []
    for batchID in range(numOfMinibatches):
        if batchID == numOfMinibatches-1:
            numbOfBatches = numOfLastMinibatch
        else:
            numbOfBatches = batchSize
        leftIndex = batchID * batchSize
        rightIndex = leftIndex + numbOfBatches
        
        
        courseid_LR =  course_id_LR[leftIndex: rightIndex].clone().long()
        videoid =  video_id[leftIndex: rightIndex].clone().long()
        continuesfeature1 =  continues_feature1[leftIndex: rightIndex].clone()  
        courseid_CNN =  course_id_CNN[leftIndex: rightIndex].clone().long()
        continuesfeature2 =  continues_feature2[leftIndex: rightIndex].clone()
        
        predictions,LR_result,CNN_result = model(courseid_LR,courseid_CNN,continuesfeature1,continuesfeature2,videoid)
        
#         predictions = torch.round(torch.flatten(predictions))
        predictions = torch.flatten(predictions)
        LR_result = torch.flatten(LR_result)
        CNN_result = torch.flatten(CNN_result)
#         print('y: ',y[leftIndex: rightIndex])
#         loss = BCrossEntropyLoss(predictions,y[leftIndex: rightIndex].float())
        loss_final = MSELoss(predictions,y[leftIndex: rightIndex].float())
        loss_lr = MSELoss(LR_result,y[leftIndex: rightIndex].float())
        loss_cnn = MSELoss(CNN_result,y[leftIndex: rightIndex].float())
#         print('loss: ',loss)
        
        loss = loss_final + loss_lr +loss_cnn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
        
        
        #testing
        
        if(batchID%100==0):
            test_numOfMinibatches = int(len(test_course_id_LR) / batchSize) + 1
            test_numOfLastMinibatch = len(test_course_id_LR) % batchSize
            results = []
            for test_batchID in range(test_numOfMinibatches):
                if test_batchID == test_numOfMinibatches-1:
                    test_numbOfBatches = test_numOfLastMinibatch
                else:
                    test_numbOfBatches = batchSize
                test_leftIndex = test_batchID * batchSize
                test_rightIndex = test_leftIndex + test_numbOfBatches
                
                
                
                test_courseid_LR =  test_course_id_LR[test_leftIndex: test_rightIndex].clone().long()
                test_videoid =  test_video_id[test_leftIndex: test_rightIndex].clone().long()
                test_continuesfeature1 =  test_continues_feature1[test_leftIndex: test_rightIndex].clone()
                test_courseid_CNN =  test_course_id_CNN[test_leftIndex: test_rightIndex].clone().long()
                test_continuesfeature2 =  test_continues_feature2[test_leftIndex: test_rightIndex].clone()
                
                test_predictions,LR_result,CNN_result = model(test_courseid_LR,test_courseid_CNN,test_continuesfeature1,test_continuesfeature2,test_videoid)
                test_predictions = torch.round(torch.flatten(test_predictions))
#                 LR_result = torch.round(torch.flatten(LR_result))
#                 CNN_result = torch.round(torch.flatten(CNN_result))
                
                results.append(test_predictions.detach().numpy().tolist())
                
                
            result = [ item for elem in results for item in elem]
#             ground_truth = ground_truth.detach().numpy().tolist()
            acc = calculate_acc(result,ground_truth)
            acc_value.append(acc)
            print('Epoch[{}/{}],loss:{:.4f},loss_final:{:.4f},loss_LR:{:.4f},loss_CNN:{:.4f},acc:{:.4f}'.format(epoach, epoach_count,loss.item(),loss_final.item(),loss_lr.item(),loss_cnn.item(),acc))

#         batchIndex = batchList[leftIndex: rightIndex]
    end = time.time()
    interval  = end-start 
    times.append(interval)   
    print('time:{:.4f}'.format(interval))



torch.save(model.state_dict(), 'lr_cnn.model')
