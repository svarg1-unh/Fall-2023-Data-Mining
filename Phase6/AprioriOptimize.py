# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from apyori import apriori

# Data Preprocessing
file_path = 'trends_by_cat.csv'
df = pd.read_csv(file_path)
groupedDf = df.groupby("location")["category"].unique().apply(list)
categories = groupedDf.tolist()

# Run 1
rules = apriori(groupedDf, min_support=0.10, min_confidence=0.60, min_lift=2, min_length = 2)
results = list(rules)

# Extract support and confidence for plot
suppAndConf = [[None for j in range(2)] for i in range(len(results))]
for i in range(0, len(results)):
    suppAndConf[i][0] = results[i][1]
    suppAndConf[i][1] = results[i][2]
for i in range(0, len(suppAndConf)):
    suppAndConf[i][1] = suppAndConf[i][1][0][2]
plotDf = pd.DataFrame(suppAndConf)
plotDf.columns = ["support", "confidence"]

print(plotDf["support"].size)
sns.scatterplot(data=plotDf, x="support", y="confidence")
plt.show()  

# Run 2
# Lower confidence
rules = apriori(groupedDf, min_support=0.10, min_confidence=0.20, min_lift=2, min_length = 2)
results = list(rules)

# Extract support and confidence for plot
suppAndConf = [[None for j in range(2)] for i in range(len(results))]
for i in range(0, len(results)):
    suppAndConf[i][0] = results[i][1]
    suppAndConf[i][1] = results[i][2]
for i in range(0, len(suppAndConf)):
    suppAndConf[i][1] = suppAndConf[i][1][0][2]
plotDf = pd.DataFrame(suppAndConf)
plotDf.columns = ["support", "confidence"]

print(plotDf["support"].size)
sns.scatterplot(data=plotDf, x="support", y="confidence")
plt.show()  


# Run 3
# Lower lift
rules = apriori(groupedDf, min_support=0.10, min_confidence=0.60, min_lift=1, min_length = 2)
results = list(rules)

# Extract support and confidence for plot
suppAndConf = [[None for j in range(2)] for i in range(len(results))]
for i in range(0, len(results)):
    suppAndConf[i][0] = results[i][1]
    suppAndConf[i][1] = results[i][2]
for i in range(0, len(suppAndConf)):
    suppAndConf[i][1] = suppAndConf[i][1][0][2]
plotDf = pd.DataFrame(suppAndConf)
plotDf.columns = ["support", "confidence"]

print(plotDf["support"].size)
sns.scatterplot(data=plotDf, x="support", y="confidence")
plt.show() 


# Run 4
# Higher Confidence
rules = apriori(groupedDf, min_support=0.10, min_confidence=0.90, min_lift=1.7, min_length = 2)
results = list(rules)

# Extract support and confidence for plot
suppAndConf = [[None for j in range(2)] for i in range(len(results))]
for i in range(0, len(results)):
    suppAndConf[i][0] = results[i][1]
    suppAndConf[i][1] = results[i][2]
for i in range(0, len(suppAndConf)):
    suppAndConf[i][1] = suppAndConf[i][1][0][2]
plotDf = pd.DataFrame(suppAndConf)
plotDf.columns = ["support", "confidence"]

print(plotDf["support"].size)
sns.scatterplot(data=plotDf, x="support", y="confidence")
plt.show() 