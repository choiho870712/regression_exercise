#!/usr/bin/env python
# coding: utf-8

# In[8]:


import json
import csv


# In[10]:


def read_json(filename):
    return json.loads(open(filename, 'r', encoding='UTF-8').read())


# In[17]:


def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
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
        for every_time in data['cwbopendata']['dataset']['location'][0]['weatherElement'][0]['time'] :     
            # 若夜晚 ,日照 = 0
            try :
                sunshine_time = every_time['weatherElement'][6]['elementValue']['value']
            except :
                sunshine_time = 0
            
            # 寫入這一行
            writer.writerow([every_time['obsTime'], 
                             every_time['weatherElement'][0]['elementValue']['value'], 
                             every_time['weatherElement'][1]['elementValue']['value'], 
                             every_time['weatherElement'][2]['elementValue']['value'], 
                             every_time['weatherElement'][3]['elementValue']['value'], 
                             every_time['weatherElement'][4]['elementValue']['value'], 
                             every_time['weatherElement'][5]['elementValue']['value'],
                             sunshine_time])


# In[12]:


data = read_json("C-B0024-002.json")


# In[13]:


data


# In[19]:


write_csv(data,"banqiao.csv")


# In[ ]:




