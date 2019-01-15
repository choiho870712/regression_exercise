#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv


# In[49]:


def read_json(filename):
    return json.loads(open(filename, 'r', encoding='UTF-8').read())

def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入第一列資料
        writer.writerow(['station', 
                         'time', 
                         'air_pressure', 
                         'temperature', 
                         'humidity', 
                         'wind_speed', 
                         'wind_direction', 
                         'rainfall',
                         'sunshine time'])
        # 寫入剩餘資料
        for every_station in data['cwbopendata']['dataset']['location'] :
            station_name = every_station['locationName']
            for every_time in every_station['weatherElement'][0]['time'] :
                if ( every_time['weatherElement'][5]['elementValue']['value'] == 'T' ) :
                    rainfall = 0.05
                else :
                    rainfall = every_time['weatherElement'][5]['elementValue']['value']
                try :
                    writer.writerow([station_name, 
                                     every_time['obsTime'], 
                                     every_time['weatherElement'][0]['elementValue']['value'], 
                                     every_time['weatherElement'][1]['elementValue']['value'], 
                                     every_time['weatherElement'][2]['elementValue']['value'], 
                                     every_time['weatherElement'][3]['elementValue']['value'], 
                                     every_time['weatherElement'][4]['elementValue']['value'], 
                                     rainfall,
                                     every_time['weatherElement'][6]['elementValue']['value']])
                except :
                    writer.writerow([station_name, 
                                     every_time['obsTime'], 
                                     every_time['weatherElement'][0]['elementValue']['value'], 
                                     every_time['weatherElement'][1]['elementValue']['value'], 
                                     every_time['weatherElement'][2]['elementValue']['value'], 
                                     every_time['weatherElement'][3]['elementValue']['value'], 
                                     every_time['weatherElement'][4]['elementValue']['value'], 
                                     rainfall,
                                     0.0])


# In[5]:


data = read_json("C-B0024-002.json")


# In[7]:


data


# In[51]:


write_csv(data,"data.csv")


# In[ ]:




