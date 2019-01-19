#!/usr/bin/env python
# coding: utf-8

# In[7]:


# library
import csv
import numpy as np
import math


# In[8]:


# read csv

data = []
# 7 elements
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
             # 7 elements
            for i in range(7) :
                data[i].append(line[i+1])
        else :
            first_line = False


# In[9]:


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
        data[5][i] = 0.0
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


# In[10]:


# feature scaling with standardization

# need to store mean and std
mean_store = []
std_store = []
data_std = []

# standardization
for element in data :
    mean = element.mean()
    std = element.std()
    data_std.append((element-mean)/std)
    mean_store.append(mean)
    std_store.append(std)


# In[11]:


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
    y_train[i%3].append(data_std[1][i])
        
    #  9hr data
    for j in range(9) :
        # 7 elements
        for k in range(7) :
            x_train[i%3][int(i/3)].append(data_std[k][i+j])

x_train = np.array(x_train)
y_train = np.array(y_train)


# In[12]:


# model 1

# model : y = b+x*w
# difference : diff = y-y_hat
# loss : L = diff**2
# gradient decent : grad_w = dL/dw = 2*diff*x  # 2 can ignore
#                   grad_b = dL/db = 2*diff    # 2 can ignore
# adagrad : ada = sqrt(sum(g**2))
# update : w_n+1 = w_n-lr*g/ada

length = len(x_train[0][0])

lr = 1 # learning rate
iteration = 10 # iteration

# 3-fold cross validation
err_sum = 0
for va in range(3) :
    # training
    w = np.zeros(length) # weight
    b = 1 # bias
    grad_w_sum = np.zeros(length)
    grad_b_sum = 0
    
    for i in range(3) :
        # ignore if it is validation set
        if i == va :
            continue
        # train it if it is not validation set
        for j in range(iteration) :
            # stocastic gradient decent
            for k in range(len(x_train[i])) :
                # testing
                y_raw = b+np.dot(x_train[i][j],w) # model
                diff = y_raw-y_train[i][j] # difference
                loss = diff**2 # loss

                # update weight
                grad_w = diff*x_train[i][j] # gradient
                grad_w_sum += grad_w**2 # sum of gradient
                ada_w = np.sqrt(grad_w_sum) #adagrad
                w = w-lr*grad_w/ada_w # update

                # update bias
                grad_b_sum += loss**2 # sum of gradient
                ada_b = np.sqrt(grad_b_sum) #adagrad
                b = b-lr*loss/ada_b # update

                if loss < 0.0000001 :
                    break
            if loss < 0.0000001 :
                break
    # testing with validation set
    y_raw = np.dot(x_train[va],w)+b # model
    diff = y_raw-y_train[va] # difference
    loss = math.sqrt(np.sum(diff**2)) # loss
    err_sum += loss
    print ('err%d = %f  ' % (va+1,loss))

print ('avg_err = %f  ' % (err_sum/3))


# In[ ]:





# In[ ]:




