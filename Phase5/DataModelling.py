import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sklearn
from sklearn import metrics
import math
import statistics
from sklearn.isotonic import IsotonicRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.cm as cm

import pycountry_convert as pc


class GraphPlotter():
    continents = ["North America", "South America", "Africa", "Asia", "Europe", "Oceania"]
    #Initialization of Object of class GraphPlotter
    def __init__(self, df):
        self.master_df = df
        self.df = self.master_df[:]
        
        #Plotting graph for each continents as we are trying to get meaningful observations from each continents
        for continental in GraphPlotter.continents:
            self.rankedCategories = []
            self.legendsColor = []
            self.colors = iter(cm.rainbow(np.linspace(0, 1, 8)))
            self.plot_linear_regression(continental)

    def linear_regression_calculations(self,x,y): 
        # Estimate Regression Coefficients 
        b = self.estimate_coef(x, y) 
        # Calculate rmse 
        residual_error = self.rmse(b,y,x)
        return b, residual_error[0]

    # Function: Calculate Regression Coefficients : b0 is Y-intercept and b1 is slope for a Regression Line b0 + b1*x  
    def estimate_coef(self,x, y): 
        # mean of x and y vector 
        m_x, m_y = np.mean(x), np.mean(y) 
        # calculating cross-deviation and deviation about x 
        SS_xy = np.sum(y*x) - self.n*m_y*m_x 
        SS_xx = np.sum(x*x) - self.n*m_x*m_x 

        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1*m_x 
        return(b_0, b_1) 
    
    # Function: Plot the scatter plot and Regression Line as per the predicted coefficients 
    # but the result is shown after plotting all the regressions
    def plot_regression_line(self,item): 
        x = item['x']
        y = item['y']
        b = item['coef_b']
        myLabel = f"{item['category']} | RMSE: ({round(item['rmse'], 2)})"
        color = next(self.colors)
        self.legendsColor.append(Line2D([0],[0],color= color, linewidth=4, label=myLabel ))
        # plotting the actual points as scatter plot 
        plt.scatter(x, y, color = color, marker = "o", s = 30) 
    
        # predicted response vector 
        y_pred = b[0] + b[1]*x 
        
        # plot the regression line 
        plt.plot(x, y_pred, color = color) 

        # Function: Calculate RMSE (Root Mean-Squared Error values)    
    def rmse(self,b,y,x):
        predict=[]
        for i in range(0,self.n):
            predict.append(b[0]+b[1]*x[i])
        predict=np.array(predict)    
        mse = metrics.mean_squared_error(y, predict)
        root_mse = math.sqrt(mse)                       # RMSE value
        nrmse = root_mse/statistics.mean(y)             # Normalized RMSE value
        return(root_mse,nrmse)

    def plot_linear_regression(self, continental):
        temp_df = self.df.copy()
        self.df = self.master_df[self.master_df['continent'] == continental]

        #Iterating over unique categories to get frequency of each categories over a period of time 
        for cat in self.df['category'].unique():
            temp_df = self.df.copy()
            temp_df = temp_df[temp_df['category'] == cat]
            temp_df = temp_df.groupby(['year','category']).size().reset_index(name='Frequency')
            self.n = np.size(temp_df['Frequency'])
            #Limiting our dataset to consider for the data with more than 4 observations 
            if self.n > 4:
                coef_b, rmse = self.linear_regression_calculations(temp_df['year'], temp_df['Frequency'])
                # considering our regressions with rmse over 0.2 as less than 0.2 is overfitted
                #while good rmse value is considered between the range of 0.2 and 0.9
                #Here as our dataset doesn't have good rmse values so only considering for any rmse greater than 0.2
                if rmse >0.2:
                    self.rankedCategories.append({"category": cat, "x": temp_df['year'], "y": temp_df['Frequency'], "coef_b": coef_b, "rmse": rmse})
        
        #Sorting categories and our legend according to rmse value
        self.rankedCategories = sorted(self.rankedCategories, key= lambda x: x['rmse'])
        self.legendsColor = sorted(self.legendsColor, key= lambda x: x['rmse'])
        #Plotting regression line for only top 5 categories (acc. to rmse value)
        for item in self.rankedCategories[0:5]:
            self.plot_regression_line(item)

        # prepare and render the scatter plot 
        plt.legend(handles = self.legendsColor)
        plt.xlabel('Year') 
        plt.ylabel('Frequency') 
        plt.title ('Category Frequency across Year in %s'%continental)     
        plt.show() 


    # def draw_polynomial_regression(self,x, y):
    #     x = x.values.reshape(1,-1)
    #     y = y.values.reshape(1,-1)
    #     polynomial = PolynomialFeatures(degree= 3)
    #     x_poly = polynomial.fit_transform(x)
    #     regressor = LinearRegression()
    #     #Train our model
    #     regressor.fit(x_poly, y)
    #     y_poly_pred = regressor.predict(x_poly)
    #     # plt.plot(x,y_poly_pred, color= next(self.colors))

    #     rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    #     print(y, y_poly_pred)
    #     if rmse >= 0 and rmse < 1.8:
    #         print(rmse)
    #         color = next(self.colors)
    #         plt.scatter(x, y, color = color)
    #         plt.plot(x,y_poly_pred, color= color)


    # def draw_isotonic_regression(self,x,y):
    #     isotonic_reg = IsotonicRegression()
    #     y_iso = isotonic_reg.fit_transform(x,y)
    #     rmse = np.sqrt(mean_squared_error(x,y_iso))
    #     print(rmse)
    #     color = next(self.colors)
    #     plt.plot(x, y, 'o', label='data', color= color) 
    #     plt.plot(x,y_iso,'-',markersize=10,label='isotonic regression' , color= color)

    # def plot_polynomial_regression(self):
    #     temp_df = self.df.copy()
    #     continent = "Europe"
    #     self.df = self.master_df[self.master_df['continent'] == continent]
    #     for cat in self.df['category'].unique():

    #         temp_df = self.df.copy()
    #         temp_df = temp_df[temp_df['category'] == cat]
    #         temp_df = temp_df.groupby(['year','category']).size().reset_index(name='Frequency')
    #         self.n = np.size(temp_df['Frequency'])
    #         if self.n > 8:
    #             self.draw_polynomial_regression(temp_df['year'], temp_df['Frequency'])
    #      # prepare and render the scatter plot 
    #     plt.xlabel('Year') 
    #     plt.ylabel('Frequency') 
    #     plt.title (f'Category Frequency across Year in {continent}')     
    #     plt.show() 

def main():
    df = pd.read_csv('./trends_by_cat.csv')

    #Removing 'Global' location as it's not specific to any location
    df = df[df['location'] != 'Global']
    df.loc[df['location'] == 'Myanmar (Burma)', 'location']= 'Myanmar'

    #Getting continent name from country name using library pycountry_convert
    continent = []
    for item in df['location']:
        country_code = pc.country_name_to_country_alpha2(item, cn_name_format="default")
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        continent.append(continent_name)

    df['continent']= continent
    plotter = GraphPlotter(df)


if __name__ == "__main__":
    main()
