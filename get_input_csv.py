#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import json


# In[3]:


# get input csv
data = json.loads(open('C-B0024-002.json', 'r', encoding='UTF-8').read())

with open('input.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入第一列資料
    writer.writerow(['time', 
                     'air_pressure', 
                     'temperature', 
                     'humidity', 
                     'wind_speed', 
                     'wind_direction', 
                     'rainfall',
                     'sunshine_time'])
    # 寫入資料
    data_len = len(data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'])
    for i in range(data_len-10,data_len-1) :
        # 若夜晚 ,日照 = 0
        try :
            sunshine_time = data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][6]['elementValue']['value']
        except :
            sunshine_time = 0

        # 寫入這一行
        writer.writerow([data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['obsTime'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][0]['elementValue']['value'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][1]['elementValue']['value'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][2]['elementValue']['value'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][3]['elementValue']['value'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][4]['elementValue']['value'], 
                         data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'][i]['weatherElement'][5]['elementValue']['value'],
                         sunshine_time])


# In[ ]:




