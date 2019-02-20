# library
import csv
import numpy as np
import math

# read model
mean = np.load('mean.npy')
std = np.load('std.npy')
w = np.load('model.npy')

# input csv file
data = []
# 7 features
for i in range(7):
    data.append([])
    
# open csv
with open('test.csv', 'r') as csvfile :
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

# for all data
for i in range(9) :
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
data_std = []
for i in range(7) :
    data_std.append((data[i]-mean[i])/std[i])

data_std = np.array(data_std)

# get x
x = []
for i in range(9) :
    # 7 features
    for j in range(7) :
        x.append(data_std[j][i])

# use model
y = np.dot(x,w) # model
y = y*std[1]+mean[1] # unscale temperature

print(y)

