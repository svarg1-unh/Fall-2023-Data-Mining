import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

countries = ["Sweden", "Turkey", "Romania", "Czech Republic", "Norway", "Italy", "Austria", "Netherlands", "Poland", "Switzerland", \
			 "France", "Finland", "Ukraine", "United Kingdom", "Denmark", "Germany", "Portugal", "Belgium"]

# Data preprocessing
trends = pd.read_csv("../trends.csv")
trends = trends.drop("country_code", axis=1)
trends = trends.drop("region_code", axis=1)
trends = trends.drop("refresh_date", axis=1)
trends.columns = ["Score", "Week", "Rank", "Country", "Region", "Term"]
# Extract month and year from week attribute
frame = pd.to_datetime(trends["Week"])
frame = pd.DataFrame([frame]).transpose() 
frame["date"] = frame
trends["Month"] = frame["date"].dt.month
trends["Year"] = frame["date"].dt.year
trends.drop("Week", axis=1)
trends = trends[["Score", "Year", "Month", "Rank", "Country", "Region", "Term"]]

# Create dataframes for Eurpean countries
europeTrends = trends[trends["Country"].isin(countries)]
swedenTrends = trends[trends["Country"] == "Sweden"]
turkeyTrends = trends[trends["Country"] == "Turkey"]
romaniaTrends = trends[trends["Country"] == "Romania"]
czechTrends = trends[trends["Country"] == "Czech Republic"]
norwayTrends = trends[trends["Country"] == "Norway"]
italyTrends = trends[trends["Country"] == "Italy"]
austriaTrends = trends[trends["Country"] == "Austria"]
dutchTrends = trends[trends["Country"] == "Netherlands"]
polandTrends = trends[trends["Country"] == "Poland"]
swissTrends = trends[trends["Country"] == "Switzerland"]
frenchTrends = trends[trends["Country"] == "France"]
finnishTrends = trends[trends["Country"] == "Finland"]
ukrainianTrends = trends[trends["Country"] == "Ukraine"]
britishTrends = trends[trends["Country"] == "United Kingdom"]
danishTrends = trends[trends["Country"] == "Denmark"]
germanTrends = trends[trends["Country"] == "Germany"]
portugalTrends = trends[trends["Country"] == "Portugal"]
belgianTrends = trends[trends["Country"] == "Belgium"]
print(europeTrends.head())
