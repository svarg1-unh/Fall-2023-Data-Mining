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




