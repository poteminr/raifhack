from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

salary_data = pd.read_csv('data/salary.csv', delimiter=";")[["Name", "2020"]]
salary_data.rename(columns={"Name":"region", '2020':'salary'}, inplace=True)

city_data = pd.read_csv('data/city.csv')
city_data.rename(columns={'geo_lat':'centre_lat', 'geo_lon':'centre_lng'}, inplace=True)
city_features = ['city', 'fias_level', 'capital_marker', 'centre_lat', 'centre_lng', 'population','federal_district']



def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371* c
    return km


def preproces_data(dataframe):
    dataframe = pd.merge(dataframe, salary_data, on='region')
    dataframe = pd.merge(dataframe, city_data[city_features], on='city')

    dataframe['distance_from_centre_in_km'] = dataframe.apply(lambda x: haversine(x.lng, x.lat, x.centre_lng, x.centre_lat), axis=1)
    dataframe['osm_city_nearest_name'].replace('Артём', 'Артем', inplace=True)
    dataframe['is_nearest_city'] = (dataframe['city'] == dataframe['osm_city_nearest_name'])


    dataframe['population'] = dataframe.apply(lambda x: x['osm_city_nearest_population'] if x['is_nearest_city'] else x['population'], axis=1)
    

    return dataframe
    
