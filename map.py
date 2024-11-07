import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
import pickle

# def convert(s):
#     s = s.replace("\n", "")
#     s = s.split("#")
#     return s

# with open("subarea.txt", "r", encoding = "utf-8") as f:
#     sub_areas = f.readlines()

def user_input_features():
    data = {}
    for col in main_cols:
        data[col] = [st.number_input("Insert a " + col, value=None, placeholder="Type a " + col + "...")]
    features = pd.DataFrame.from_dict(data)
    return features

gdf = gpd.read_file('mo.geojson')
gdf['Center_point'] = gdf['geometry'].centroid
gdf["lon"] = gdf.Center_point.map(lambda p: p.x)
gdf["lat"] = gdf.Center_point.map(lambda p: p.y)
gdf = pd.DataFrame(gdf, columns = ["lon", 'lat', 'NAME'])
# d = dict(list(map(convert, sub_areas)))
# gdf['NAME'] = gdf['NAME'].replace(d)

main_cols = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'kitch_sq', 'metro_min_avto']
# df = pd.read_csv("train.csv", usecols = main_cols)
# t.rename(columns={'sub_area':'NAME'}, inplace=True)
# df = pd.merge(df, t, on = "NAME", how = "inner") 

model = pickle.load(open("rmsle.dat", "rb"))
st.header("Предсказание цен на недвижимость", divider="gray")
st.image("kreml.png")
st.map(gdf)
price = model.predict(user_input_features())[0]
st.write(price)
