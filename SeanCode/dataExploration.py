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
trends.columns = ["Score", "Year-Month", "Rank", "Country", "Sub-Region", "Term"]
trends = trends[["Term", "Rank", "Country", "Sub-Region", "Year-Month", "Score"]]

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

# Assign region classifiers to countries
europeTrends = europeTrends.assign(Region = europeTrends["Country"])
europeTrends.loc[europeTrends["Region"].isin(nordicCountries), "Region"] = "Nordic Countries"
europeTrends.loc[europeTrends["Region"].isin(eastEuroCountries), "Region"] = "Eastern Europe"
europeTrends.loc[europeTrends["Region"].isin(southEuroCountries), "Region"] = "Southern Europe"
europeTrends.loc[europeTrends["Region"].isin(centralEuroCountries), "Region"] = "Central Europe"
europeTrends.loc[europeTrends["Region"].isin(westEuroCountries), "Region"] = "Western Europe"
