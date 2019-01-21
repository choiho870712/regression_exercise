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

# data preprocessing ///////////////////////////////////////

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
for i in range(3) :
    x_train.append([])
    y_train.append([])

# 8776 kinds of 10hr data
for i in range(8775) :
    x_train[i%3].append([])
    y_train[i%3].append(data[1][i+9])
        
    #  9hr data
    for j in range(9) :
        # 7 features
        for k in range(7) :
            x_train[i%3][int(i/3)].append(data_std[k][i+j])

x_train = np.array(x_train)
y_train = np.array(y_train)

# training ////////////////////////////////////////////////////
def training() :
    lr = 1 # learning rate

    try :
        cur_w = np.load('weight.npy') # weight
        cur_b = np.load('bias.npy') # bias
        best_w = cur_w # weight
        best_b = cur_b # bias
        best_avg_loss = np.load('best_avg_loss.npy')
        grad_w_sum = np.load('grad_w_sum.npy')
        grad_b_sum = np.load('grad_b_sum.npy')
    except :
        cur_w = np.zeros(len(x_train[0][0])) # weight
        cur_b = 1.0 # bias
        best_w = np.zeros(len(x_train[0][0])) # weight
        best_b = 1.0 # bias
        best_avg_loss = 1.0
        grad_w_sum = np.zeros(len(x_train[0][0]))
        grad_b_sum = 0.0

    iteration = 0
    while True :
        iteration += 1
        # training with full data
        for i in range(3) :
            # stocastic gradient decent
            for j in range(len(x_train[i])) :
                # testing
                y_raw = model(cur_b,cur_w,x_train[i][j]) # model
                diff = difference(y_raw,y_train[i][j]) # difference
                loss = loss_function(diff) # loss

                # update weight
                grad_w = grad_w_function(diff,x_train[i][j]) # gradient
                grad_w_sum += grad_w**2 # sum of gradient
                ada_w = np.sqrt(grad_w_sum) #adagrad
                cur_w = cur_w-lr*grad_w/ada_w # update

                # update bias
                grad_b = grad_b_function(diff) # gradient
                grad_b_sum += grad_b**2 # sum of gradient
                ada_b = np.sqrt(grad_b_sum) #adagrad
                cur_b = cur_b-lr*grad_b/ada_b # update
        
        # 3-fold cross validation testing average loss
        loss_sum = 0.0
        for va in range(3) :
            # testing with validation set
            y_raw = model(cur_b,cur_w,x_train[va]) # model
            diff = difference(y_raw,y_train[va]) # difference
            loss = np.sum(loss_function(diff))/len(diff) # loss
            loss_sum += loss

        cur_avg_loss = loss_sum/3
        print('avg_loss = %f    ' % (cur_avg_loss))
        
        # store weight and bias if it has best average loss
        if cur_avg_loss < best_avg_loss :
            best_avg_loss = cur_avg_loss
            best_w = cur_w
            best_b = cur_b
    
        # store model while 10 iteration
        if (iteration%10) == 0 :
            np.save('weight.npy', best_w)
            np.save('bias.npy', best_b)
            np.save('grad_w_sum.npy',grad_w_sum)
            np.save('grad_b_sum.npy',grad_b_sum)
            np.save('best_avg_loss.npy',best_avg_loss)
            print('saved model')

# modeling ///////////////////////////////////////////////

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

def grad_b_function(diff) :
    return diff

training()
 