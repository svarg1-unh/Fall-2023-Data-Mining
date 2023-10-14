# Categorical - Categorical
#Importing the necessary modules
import matplotlib.pyplot as plt
import pandas as pd


#Initializing the lists for X and Y
data = pd.read_csv('/Users/rajdeepbhattacharya/Desktop/Phase 4-Data Exploration-Assignment/trends-rankterm.csv')

df = pd.DataFrame(data)

X = list(df.iloc[:, 1])
Y = list(df.iloc[:, 0])

#Plotting the data using bar() method
plt.bar(X, Y, color='g')
plt.title("Bar Graph On Terms And Ranks")
plt.xlabel("Terms")
plt.ylabel("Ranks")

#Showing the plot
plt.show()






# Numerical-Numerical
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./trends.csv')
df = df.drop(columns = ['region_name', 'region_code', 'score', 'refresh_date'], axis=1)
df['week'] = df['week'].str[:7]
df = df.rename(columns={'week':'year'})

newDf = df.groupby(['year', 'term']).size().reset_index(name='Frequency')
newDf = newDf[newDf['Frequency'] > 6]

print(newDf)
print(newDf['Frequency'].value_counts())
sns.relplot( data= newDf[['year','term', 'Frequency']], x = 'year', y = 'Frequency', hue = 'term')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.title('Most Significant Search Terms')
plt.xticks(rotation='vertical', fontsize=8)
plt.tight_layout()
plt.show()

liverpoolDf = newDf[newDf['term'] == 'Liverpool']
print(liverpoolDf)
#Demonstrating Pearson's correlation coefficient between Date and search Term Frequency of Liverpool
liverpoolDf['Numeric_date'] = pd.to_datetime(liverpoolDf['year']).apply(lambda x: x.timestamp())

#Create Pearson's correlation coefficient between Date and search Term Frequency of Liverpool
correlation = liverpoolDf['Numeric_date'].corr(liverpoolDf['Frequency'])
print(f"Correlation: {correlation: .2f} ")

plt.figure(figsize=(10, 6))
plt.plot(liverpoolDf['year'], liverpoolDf['Frequency'], label='Frequency')
plt.xlabel('Date (Timestamp)')
plt.ylabel('Frequency')
plt.title('Search Term Frequency Over Time of \'Liverpool\'')
plt.legend()

# Display the plot
plt.text(0.1, 0.9, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, color='red')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()






# Categorical-Numerical
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
turkeyTrends = europeTrends[europeTrends["Country"] == "Turkey"]

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

# Plot number of searches in Turkey by top 10 regions
turkeyTrendsCount = turkeyTrends.groupby("Sub-Region", as_index=False).count()
sns.barplot(x="Sub-Region", y="Term", data=turkeyTrendsCount.nlargest(10, "Term"))
plt.xlabel("")
plt.ylabel("Searches")
plt.title("Number of Searches by Top 10 Turkish Regions")
plt.show()

# Plot number of searches in Turkey by bottom 10 regions
sns.barplot(x="Sub-Region", y="Term", data=turkeyTrendsCount.nsmallest(10, "Term"))
plt.xlabel("")
plt.ylabel("Searches")
plt.title("Number of Searches by Bottom 10 Turkish Regions")
plt.show()