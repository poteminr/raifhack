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

def calculate_statistic(dataframe):
    dataframe['population_per_house_1000'] = dataframe['reform_house_population_1000'] / dataframe['reform_count_of_houses_1000']
    dataframe['population_per_house_500'] = dataframe['reform_house_population_500'] / dataframe['reform_count_of_houses_500']

    dataframe['population_per_floor_1000'] = dataframe['population_per_house_1000'] / dataframe['reform_mean_floor_count_1000']
    dataframe['population_per_floor_500'] = dataframe['population_per_house_500'] / dataframe['reform_mean_floor_count_500']

    dataframe['mean_house_age_500'] = 2020 - dataframe['reform_mean_year_building_500']
    dataframe['mean_house_age_1000'] = 2020 - dataframe['reform_mean_year_building_1000']


    radius_buildings_features = ['osm_catering_points_in_', 'osm_shops_points_in_', 'osm_offices_points_in_', 'osm_finance_points_in_', 
                       'osm_healthcare_points_in_', 'osm_leisure_points_in_', 'osm_hotels_points_in_', 'osm_amenity_points_in_']
    
    population_features = radius_buildings_features + ['osm_leisure_points_in_', 'osm_culture_points_in_', 'osm_train_stop_points_in_',
                                                       'osm_transport_stop_points_in_', 'osm_crossing_points_in_']

    col_list = dataframe.columns
    # for radius in ['0.001', '0.005', '0.0075', '0.01']:
    #     for feature in radius_buildings_features:
    #         col_name = feature + radius
    #         if col_name in col_list:
    #             dataframe[f'percent of {col_name}'] = dataframe[col_name] / dataframe[f'osm_building_points_in_{radius}']

    for radius in ['0.005', '0.01']:
        for feature in population_features:
            col_name = feature + radius
            if col_name in col_list:

                dataframe[f'{col_name} per population'] = dataframe[col_name] / dataframe[f'reform_house_population_{int(float(radius)*100000)}'] 

    return dataframe


def preproces_data(dataframe):
    dataframe = pd.merge(dataframe, salary_data, on='region')
    dataframe = pd.merge(dataframe, city_data[city_features], on='city')

    dataframe['distance_from_centre_in_km'] = dataframe.apply(lambda x: haversine(x.lng, x.lat, x.centre_lng, x.centre_lat), axis=1)
    dataframe['osm_city_nearest_name'].replace('Артём', 'Артем', inplace=True)
    dataframe['is_nearest_city'] = (dataframe['city'] == dataframe['osm_city_nearest_name'])


    dataframe['population'] = dataframe.apply(lambda x: x['osm_city_nearest_population'] if x['is_nearest_city'] else x['population'], axis=1)
    

    return dataframe
    
