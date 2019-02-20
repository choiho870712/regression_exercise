# library
import csv
import numpy as np
import math

# read csv ////////////////////////////////////////////

data = []
# 7 features
for i in range(7):
	data.append([])
    
# open csv
with open('train.csv', 'r') as csvfile :
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

# data preprocessing ///////////////////////////////////////

data_len = len(data[0])

# for all data
for i in range(data_len) :
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
    data[3][i] = wind_speed*math.sin(wind_direction/180*math.pi)
    data[4][i] = wind_speed*math.cos(wind_direction/180*math.pi)
    
data = np.array(data)

# feature scaling with standardization
mean_store = []
std_store = []
data_std = []
for feature in data :
    mean = feature.mean()
    std = feature.std()
    data_std.append((feature-mean)/std)
    mean_store.append(mean)
    std_store.append(std)

# store mean and std
mean_store = np.array(mean_store)
std_store = np.array(std_store)
np.save('mean.npy', mean_store)
np.save('std.npy', std_store)

# get training data in 3 parts ////////////////////////////////
x_train = []
y_train = []

# 8776 kinds of 10hr data
for i in range(data_len-9) :
    x_train.append([])
    y_train.append(data_std[1][i+9])
        
    #  9hr data
    for j in range(9) :
        # 7 features
        for k in range(7) :
            x_train[i].append(data_std[k][i+j])

x_train = np.array(x_train)
y_train = np.array(y_train)

# training ////////////////////////////////////////////////////
lr = 0.0001 # learning rate
iteration = 10
w = np.zeros(63) # weight
ada = np.zeros(63)

for i in range(iteration) :
    # stocastic gradient decent
    for j in range(data_len-9) :
        y = np.dot(x_train[j],w) # model
        diff = y - y_train[j] # difference
        ada += diff # adagrad
        grad = diff*x_train[j] # gradient
        w = w + lr*grad/ada # update
        print(diff)

np.save('model.npy', w)
 