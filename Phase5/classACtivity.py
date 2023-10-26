import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import math
import statistics

import pycountry_convert as pc

# Function: Call the functions in a particular order
def draw_linear_regression(x,y): 
    # Estimate Regression Coefficients 

    b = estimate_coef(x, y) 
    print("Estimated coefficients of the line y = b0 + b1*x are:\nb0 = {}   \nb1 = {}".format(b[0], b[1])) 
      
    # Plot regression line 
    residual_error = rmse(b,y,x)
    print("RMSE VALUE is",residual_error[0])
    print("Normalized RMSE VALUE is",residual_error[1])
    plot_regression_line(x, y, b)

 # Function: Calculate Regression Coefficients : b0 is Y-intercept and b1 is slope for a Regression Line b0 + b1*x  
def estimate_coef(x, y): 
     
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
   
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
# Function: Plot the scatter plot and Regression Line as per the predicted coefficients
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
      
    # plot the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # prepare and render the scatter plot 
    plt.xlabel('Year') 
    plt.ylabel('Global Average CO2 Concentrations (ppm)') 
    plt.title ('Yearly Global Average CO2 Concentrations in parts per million (ppm) and Linear Regression')     
    plt.show() 

    # Function: Calculate RMSE (Root Mean-Squared Error values)    
def rmse(b,y,x):
    predict=[]
    for i in range(0,n):
        predict.append(b[0]+b[1]*x[i])
    predict=np.array(predict)    
    mse = metrics.mean_squared_error(y, predict)
    root_mse = math.sqrt(mse)                       # RMSE value
    nrmse = root_mse/statistics.mean(y)             # Normalized RMSE value
    return(root_mse,nrmse)


df = pd.read_csv('./trends_by_cat.csv')

#Removing 'Global' location 
df = df[df['location'] != 'Global']
df.loc[df['location'] == 'Myanmar (Burma)', 'location']= 'Myanmar'

#Filtering irrelevant category


continent = []
for item in df['location']:

    country_code = pc.country_name_to_country_alpha2(item, cn_name_format="default")
    # print(country_code)
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    continent.append(continent_name)

df['continent']= continent
# print(df.head(10))
newDf = df[df['category'] == 'People']
peopleData = newDf.groupby(['year','category']).size().reset_index(name='Frequency')
# newDf = newDf[newDf['Frequency'] > 6]
# print(newDf.sort_values('Frequency', ascending= False))

# peopleData = newDf[newDf['category'] == 'People']
print(peopleData)
n = np.size(peopleData['Frequency'])  # number of observations/points 
draw_linear_regression(peopleData['year'], peopleData['Frequency'])