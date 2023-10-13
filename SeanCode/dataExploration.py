import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Create dataframes for European countries
europeTrends = trends[trends["Country"].isin(countries)]
nordicTrends = europeTrends[europeTrends["Country"].isin(nordicCountries)]
eastEuroTrends = europeTrends[europeTrends["Country"].isin(eastEuroCountries)]
southEuroTrends = europeTrends[europeTrends["Country"].isin(southEuroCountries)]
centralEuroTrends = europeTrends[europeTrends["Country"].isin(centralEuroCountries)]
westEuroTrends = europeTrends[europeTrends["Country"].isin(westEuroCountries)]
swedenTrends = europeTrends[europeTrends["Country"] == "Sweden"]
turkeyTrends = europeTrends[europeTrends["Country"] == "Turkey"]
romaniaTrends = europeTrends[europeTrends["Country"] == "Romania"]
czechTrends = europeTrends[europeTrends["Country"] == "Czech Republic"]
norwayTrends = europeTrends[europeTrends["Country"] == "Norway"]
italyTrends = europeTrends[europeTrends["Country"] == "Italy"]
austriaTrends = europeTrends[europeTrends["Country"] == "Austria"]
dutchTrends = europeTrends[europeTrends["Country"] == "Netherlands"]
polandTrends = europeTrends[europeTrends["Country"] == "Poland"]
swissTrends = europeTrends[europeTrends["Country"] == "Switzerland"]
frenchTrends = europeTrends[europeTrends["Country"] == "France"]
finnishTrends = europeTrends[europeTrends["Country"] == "Finland"]
ukrainianTrends = europeTrends[europeTrends["Country"] == "Ukraine"]
britishTrends = europeTrends[europeTrends["Country"] == "United Kingdom"]
danishTrends = europeTrends[europeTrends["Country"] == "Denmark"]
germanTrends = europeTrends[europeTrends["Country"] == "Germany"]
portugalTrends = europeTrends[europeTrends["Country"] == "Portugal"]
belgianTrends = europeTrends[europeTrends["Country"] == "Belgium"]

# Assign region classifiers to countries
europeTrends = europeTrends.assign(Region = europeTrends["Country"])
europeTrends.loc[europeTrends["Region"].isin(nordicCountries), "Region"] = "Nordic Countries"
europeTrends.loc[europeTrends["Region"].isin(eastEuroCountries), "Region"] = "Eastern Europe"
europeTrends.loc[europeTrends["Region"].isin(southEuroCountries), "Region"] = "Southern Europe"
europeTrends.loc[europeTrends["Region"].isin(centralEuroCountries), "Region"] = "Central Europe"
europeTrends.loc[europeTrends["Region"].isin(westEuroCountries), "Region"] = "Western Europe"

# Plot number of searches by region
europeTrendsCount = europeTrends.groupby("Region", as_index=False).count()
sns.barplot(x="Region", y="Term", data=europeTrendsCount)
plt.xlabel("")
plt.ylabel("Searches")
plt.title("Number of Searches by Region")
plt.show()

# Plot number of searches in Southern Europe by country
southEuroTrendsCount = southEuroTrends.groupby("Country", as_index=False).count()
sns.barplot(x="Country", y="Term", data=southEuroTrendsCount)
plt.xlabel("")
plt.ylabel("Searches")
plt.title("Number of Searches by Southern European Country")
plt.show()