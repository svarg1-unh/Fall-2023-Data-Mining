import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data preprocessing
trends = pd.read_csv("../trends.csv")
trends = trends.drop("country_code", axis=1)
trends = trends.drop("region_code", axis=1)
trends = trends.drop("refresh_date", axis=1)
trends.columns = ["Score", "Week", "Rank", "Country", "Region", "Term"]

frame = pd.to_datetime(trends["Week"])
frame = pd.DataFrame([frame]).transpose() 
frame["date"] = frame
trends["Month"] = frame["date"].dt.month
trends["Year"] = frame["date"].dt.year
trends.drop("Week", axis=1)
trends = trends[["Score", "Year", "Month", "Rank", "Country", "Region", "Term"]]
print(trends.head(5))
print(trends["Country"].unique())

