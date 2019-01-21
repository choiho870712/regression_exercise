# regression_exercise(temperature prediction)

raw data : https://opendata.cwb.gov.tw/dataset/climate/C-B0024-002

regression.docx : the "how to do" note

banqiao.csv : weather data in a year, there are 8 features :

    1. time

    2. air_pressure

    3. temperature

    4. humidity

    5. wind_speed

    6. wind_direction

    7. rainfall

    8. sunshine_time


json_to_csv.py : the program transfer raw data(.json) to training data(banqiao.csv)

training.py : the program input training data and then train the temperature prediction model

get_input_csv.py : the program getting source input for the prediction

using_model.py : the program using input.csv and models to try predicting temperature

.npy file : model file
