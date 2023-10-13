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
nordicCountries = ["Sweden", "Norway", "Denmark", "Finland"]
eastEuroCountries = ["Romania", "Poland", "Ukraine"]
southEuroCountries = ["Turkey", "Italy", "Portugal"]
centralEuroCountries = ["Czech Republic", "Germany", "Austria", "Switzerland"]
westEuroCountries = ["Netherlands", "France", "United Kingdom", "Belgium"]

# Data preprocessing
trends = pd.read_csv("../trends.csv")
trends = trends.drop(columns = ["country_code", "region_code", "refresh_date"], axis=1)
trends["week"] = trends["week"].str[:7]
trends = trends.rename(columns = {"week":"Year-Month"})
trends.columns = ["Score", "Year-Month", "Rank", "Country", "Region", "Term"]
trends = trends[["Term", "Rank", "Country", "Region", "Year-Month", "Score"]]

# Create dataframes for Eurpean countries
europeTrends = trends[trends["Country"].isin(countries)]
nordicTrends = trends[trends["Country"].isin(nordicCountries)]
eastEuroTrends = trends[trends["Country"].isin(eastEuroCountries)]
southEuroTrends = trends[trends["Country"].isin(southEuroCountries)]
centralEuroTrends = trends[trends["Country"].isin(centralEuroCountries)]
westEuroTrends = trends[trends["Country"].isin(westEuroCountries)]
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

#europeTopTen = europeTrends[europeTrends["Rank"] <= 10]
#europe2023 = europeTrends[europeTrends["Year"] == 2023]
#europe2023TopTen = europe2023[europe2023["Rank"] <= 10]
print(len(europeTrends["Term"].unique()))
print(len(nordicTrends["Term"].unique()))
print(len(eastEuroTrends["Term"].unique()))
print(len(southEuroTrends["Term"].unique()))
print(len(centralEuroTrends["Term"].unique()))
print(len(westEuroTrends["Term"].unique()))