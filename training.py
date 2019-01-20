#!/usr/bin/env python
# coding: utf-8

# In[2]:


# library
import csv
import numpy as np
import math


# In[3]:


# read csv

data = []
# 7 features
for i in range(7):
	data.append([])
    
# open csv
with open('banqiao.csv', 'r') as csvfile :
    # csv reader
    reader = csv.reader(csvfile ,delimiter=",")
    
    # process with line
    first_line = True
    for line in reader :
        # except first line
        if first_line != True :
             # 7 features
            for i in range(7) :
                data[i].append(line[i+1])
        else :
            first_line = False


# In[4]:


# data preprocessing

# for all data
for i in range(8784) :
    # string to float
    data[0][i] = float(data[0][i]) # air pressure
    data[1][i] = float(data[1][i]) # temperature
    data[2][i] = float(data[2][i]) # humidity
    data[6][i] = float(data[6][i]) # sunshine time
    
    # rainfall
    if ( data[5][i] == 'T' ) :
        data[5][i] = 0.05
    else :
        data[5][i] = float(data[5][i])
    
    # wind speed
    wind_speed = float(data[3][i])
    
    # wind direction to polar form
    wind_direction = 0.0
    raw_wind_direction = data[4][i]
    if raw_wind_direction == '北,N' :
        wind_direction = 0.0
    elif raw_wind_direction == '北北東,NNE' :
        wind_direction = 22.5
    elif raw_wind_direction == '東北,NE' :
        wind_direction = 45.0
    elif raw_wind_direction == '東北東,ENE' :
        wind_direction = 67.5
    elif raw_wind_direction == '東,E' :
        wind_direction = 90.0
    elif raw_wind_direction == '東南東,ESE' :
        wind_direction = 112.5
    elif raw_wind_direction == '東南,ES' :
        wind_direction = 135.0
    elif raw_wind_direction == '南南東,SSE' :
        wind_direction = 157.5
    elif raw_wind_direction == '南,S' :
        wind_direction = 180.0
    elif raw_wind_direction == '南南西,SSW' :
        wind_direction = 202.5
    elif raw_wind_direction == '西南,WS' :
        wind_direction = 225.0
    elif raw_wind_direction == '西南西,WSW' :
        wind_direction = 247.5
    elif raw_wind_direction == '西,W' :
        wind_direction = 270.0
    elif raw_wind_direction == '西北西,WNW' :
        wind_direction = 292.5
    elif raw_wind_direction == '西北,WN' :
        wind_direction = 315.0
    elif raw_wind_direction == '北北西,NNW' :
        wind_direction = 337.5
    elif raw_wind_direction == '靜風,Calm' :
        wind_direction = 0
    elif raw_wind_direction == '風向不定,Variable' :
        wind_direction = 0
    else :
        print(raw_wind_direction)

    # wind to rectangular form ( wind = wind_speed * wind_direction )
    data[3][i] = wind_speed*math.sin(wind_direction/360)
    data[4][i] = wind_speed*math.cos(wind_direction/360)
    
data = np.array(data)


# In[5]:


# feature scaling with standardization

# need to store mean and std
mean_store = []
std_store = []
data_std = []

# standardization
for feature in data :
    mean = feature.mean()
    std = feature.std()
    data_std.append((feature-mean)/std)
    mean_store.append(mean)
    std_store.append(std)


# In[6]:


# get training data and testing data

x_train = []
y_train = []

 # devide training data in 3 parts
for i in range(3) :
    x_train.append([])
    y_train.append([])

# 8776 kinds of 10hr data
for i in range(8775) :
    x_train[i%3].append([])
    y_train[i%3].append(data_std[1][i+9])
        
    #  9hr data
    for j in range(9) :
        # 7 features
        for k in range(7) :
            x_train[i%3][int(i/3)].append(data_std[k][i+j])

x_train = np.array(x_train)
y_train = np.array(y_train)


# In[ ]:


# training
def training(lr,iteration,breaking_point) :
    # 3-fold cross validation
    loss_sum = 0
    for va in range(3) :
        # training
        w = np.zeros(len(x_train[0][0])) # weight
        b = 1 # bias
        grad_w_sum = np.zeros(len(x_train[0][0]))
        grad_b_sum = 0

        for i in range(3) :
            # ignore if it is validation set
            if i == va :
                continue
            # repeat training same packet
            for j in range(iteration) :
                # stocastic gradient decent
                for k in range(len(x_train[i])) :
                    # testing
                    y_raw = model(b,w,x_train[i][k]) # model
                    diff = difference(y_raw,y_train[i][k]) # difference
                    loss = loss_function(diff) # loss

                    # update weight
                    grad_w = grad_w_function(diff,x_train[i][k]) # gradient
                    grad_w_sum += grad_w**2 # sum of gradient
                    ada_w = np.sqrt(grad_w_sum) #adagrad
                    w = w-lr*grad_w/ada_w # update

                    # update bias
                    grad_b = grad_b_function(diff,x_train[i][k]) # gradient
                    grad_b_sum += grad_b**2 # sum of gradient
                    ada_b = np.sqrt(grad_b_sum) #adagrad
                    b = b-lr*grad_b/ada_b # update

                    if loss < breaking_point :
                        break
                if loss < breaking_point :
                    break
        # testing with validation set
        y_raw = model(b,w,x_train[va]) # model
        diff = difference(y_raw,y_train[va]) # difference
        loss = np.sum(loss_function(diff))/len(diff) # loss
        loss_sum += loss
        
    avg_loss = loss_sum/3
    return avg_loss


# In[ ]:


# modeling 1

# model : y = b+x*w
# difference : diff = y-y_hat
# loss : L = diff**2
# gradient decent : grad_w = dL/dw = diff*x
#                   grad_b = dL/db = diff
# adagrad : ada = sqrt(sum(g**2))
# update : w_n+1 = w_n-lr*g/ada

def model(b,w,x) :
    return np.dot(x,w)+b

def difference(y,y_hat) :
    return y-y_hat
    
def loss_function(diff) :
    return diff**2

def grad_w_function(diff,x) :
    return diff*x

def grad_b_function(diff,x) :
    return diff

lr = 1 # learning rate
breaking_point = 0.00000000000001 # when to stop

# find the best iteration
best_avg_loss = 1.0
best_iteration = 0
for iteration in range(1000) :
    avg_loss = training(lr,iteration,breaking_point)
    print('avg_loss%d = %f  ' % (iteration+1,avg_loss))
    if avg_loss < best_avg_loss :
        best_avg_loss = avg_loss
        best_iteration = iteration

print ('best_iteration = %d best_avg_loss = %f  ' % (best_iteration,best_avg_loss))
# avg_loss = 0.018675   >> not overfitting
# breaking point hitted >> not underfitting
# it's a good model, use it


# In[ ]:




