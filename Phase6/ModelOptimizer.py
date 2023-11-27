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
from sklearn.model_selection import train_test_split

import pycountry_convert as pc


class GraphPlotter():
    continents = ["North America", "South America", "Africa", "Asia", "Europe", "Oceania"]
    #Initialization of Object of class GraphPlotter
    def __init__(self, df):
        self.master_df = df
        self.df = self.master_df[:]
        self.filteredCategories = []
        
        #Plotting graph for each continents as we are trying to get meaningful observations from each continents
        # for continental in GraphPlotter.continents:
        self.rankedCategories = []
        self.legendsColor = []
        self.colors = iter(cm.rainbow(np.linspace(0, 1, 8)))
        self.plot_linear_regression()

        #Plotting graph for each continents as we are trying to get meaningful observations from each continents
        # for continental in GraphPlotter.continents:
        self.legendsColor = []
        self.colors = iter(cm.rainbow(np.linspace(0, 1, 8)))
        # self.isotonic_plotter()


        self.gradient_descent()
        

    def linear_regression_calculations(self,x,y, x2, y2): 
        # Estimate Regression Coefficients 
        self.b = self.estimate_coef(x, y) 
        # Calculate rmse 
        residual_error = self.rmse(self.b,np.array(y2),np.array(x2))
        return self.b, residual_error[0]

    # Function: Calculate Regression Coefficients : b0 is Y-intercept and b1 is slope for a Regression Line b0 + b1*x  
    def estimate_coef(self,x, y): 
        # mean of x and y vector 
        m_x, m_y = np.mean(x), np.mean(y) 
        # calculating cross-deviation and deviation about x 
        SS_xy = np.sum(y*x) - self.n*m_y*m_x 
        SS_xx = np.sum(x*x) - self.n*m_x*m_x 

        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1*m_x 
        return [b_0, b_1]
    
    #Get Y predict for linear regression
    def y_predict(self,x, b):
        predict=[]
        for i in range(0,len(x)):
            predict.append(b[0]+b[1]*x[i])
        predict=np.array(predict)
        return predict

    def accuracy(self, y, y_pred):
        n = len(y_pred)
        return 1 - sum(abs(y_pred[i] - y[i])/y[i] for i in range(n) if y[i] != 0)/n
    
    def compute_cost(self, y, y_pred):
        n = len(y)
        return (1/ (2*n)) * (np.sum((y_pred - y)**2))

    def plot_best_plot(self, y_pred, fig, color):
        # f = plt.figure(fig)
        plt.scatter(self.x_train, self.y_train, marker= "o", color = color)
        plt.scatter(self.x_test, self.y_test,marker= "x", color = color)
        plt.plot(self.x_train, y_pred, color = color)
        plt.xlim(2000, 2024)
        plt.ylim(0, 30)
        plt.xlabel('Year') 
        plt.ylabel('Frequency') 
        plt.show()

    def update_coefficient(self, learning_rate):
        y_pred = self.y_predict(self.x_test.tolist(), self.b)
        n = len(y_pred)
        self.b[0] = self.b[0] - (learning_rate * ((1/n) * np.sum(y_pred - self.y_test.tolist())))
        self.b[1] = self.b[1] - (learning_rate * ((1/n) * np.sum(y_pred - self.y_test.tolist())))

    # Function: Plot the scatter plot and Regression Line as per the predicted coefficients 
    # but the result is shown after plotting all the regressions
    def plot_regression_line(self,item): 
        x_train = item['x_train']
        y_train = item['y_train']
        x_test = item['x_test']
        y_test = item['y_test']
        b = item['coef_b']
        myLabel = f"{item['category']} | RMSE: ({round(item['rmse'], 2)})"

        color = next(self.colors)
        self.legendsColor.append(Line2D([0],[0],color= color, linewidth=4, label=myLabel ))
        # plotting the actual points as scatter plot 
        plt.scatter(x_train, y_train, color = color, marker = "o", s = 30) 
        # predicted response vector for training data
        y_train_pred = b[0] + b[1]*np.array(x_train)
        # plot the regression line 
        plt.plot(x_train, y_train_pred, color = color)

        #Scatter plotting test data
        plt.scatter(x_test, y_test, color = color, marker = "x", s = 30)  

    # Function: Calculate RMSE (Root Mean-Squared Error values)    
    def rmse(self,b,y,x):
        predict= self.y_predict(x,b)   
        mse = metrics.mean_squared_error(y, predict)
        root_mse = math.sqrt(mse)                       # RMSE value
        nrmse = root_mse/statistics.mean(y)             # Normalized RMSE value
        return(root_mse,nrmse)

    def plot_linear_regression(self, continental=""):
        # self.df = self.master_df[self.master_df['continent'] == continental]

        #Iterating over unique categories to get frequency of each categories over a period of time 
        for cat in self.df['category'].unique():
            temp_df = self.df.copy()
            temp_df = temp_df[temp_df['category'] == cat]
            temp_df = temp_df.groupby(['year','category']).size().reset_index(name='Frequency')

            self.n = np.size(temp_df['Frequency'])
            #Limiting our dataset to consider for the data with more than 4 observations 
            if self.n > 6:
                #Splitting data into train and test data
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(temp_df['year'], temp_df['Frequency'], train_size=0.8, shuffle= True)
                
                self.n = len(self.x_train)
                coef_b, rmse = self.linear_regression_calculations(self.x_train, self.y_train, self.x_test, self.y_test)
                # considering our regressions with rmse over 0.2 as less than 0.2 is overfitted
                #while good rmse value is considered between the range of 0.2 and 0.9
                #Here as our dataset doesn't have good rmse values so only considering for any rmse greater than 0.2
                if rmse > 0.2:
                    self.rankedCategories.append({
                                                "category": cat, 
                                                "x_train": self.x_train.tolist(), 
                                                "y_train": self.y_train.tolist(), 
                                                "x_test": self.x_test.tolist(), 
                                                "y_test": self.y_test.tolist(), 
                                                "coef_b": coef_b, 
                                                "rmse": rmse
                                                })
        
        #Sorting categories and our legend according to rmse value
        self.rankedCategories = sorted(self.rankedCategories, key= lambda x: x['rmse'])
        self.legendsColor = sorted(self.legendsColor, key= lambda x: x['rmse'])
        #Plotting regression line for only top 5 categories (acc. to rmse value)
        for item in self.rankedCategories[0:4]:
            self.filteredCategories.append(item['category'])
            self.plot_regression_line(item)

        # prepare and render the scatter plot 
        plt.legend(handles = self.legendsColor)
        plt.xlabel('Year') 
        plt.ylabel('Frequency') 
        plt.title ('Linear Regression of Category Frequency across Year %s'%continental)     
        plt.show() 

        

    def gradient_descent(self):
        best_model = None
        best_accuracy = 0

        self.colors = iter(cm.rainbow(np.linspace(0, 1, 8)))
        # #Setting filtered data for gradient descent
        # # self.df = self.df.copy()
        # mast_df = self.df[self.df['category'].isin(self.filteredCategories) ]
        # print(mast_df)

        #Iterating over unique categories to get frequency of each categories over a period of time 
        for cat in self.df['category'].unique():
            
            temp_df = self.df.copy()
            temp_df = temp_df[temp_df['category'] == cat]
            temp_df = temp_df.groupby(['year','category']).size().reset_index(name='Frequency')
            
            self.n = np.size(temp_df['Frequency'])
            #Limiting our dataset to consider for the data with more than 4 observations 
            if cat in self.filteredCategories:
            # if self.n > 10:
                iterations = 10000
                steps = 50
                learning_rate = 0.01
                costs = []
                self.colors = iter(cm.rainbow(np.linspace(0, 1, 8)))
                #Splitting data into train and test data
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(temp_df['year'], temp_df['Frequency'], train_size=0.8, shuffle= True)
                filteredItems = [item for item in self.rankedCategories if item['category'] == cat ]
                b = [0,0]
                if filteredItems:
                    b = filteredItems[0]['coef_b']
                first_y_pred = self.y_predict(self.x_train.tolist(), b)

                """Setting the current accuracy before training model in gradient descent as best accuracy
                """
                test_pred = self.y_predict(self.x_test.tolist(), b)
                best_accuracy = self.accuracy(self.y_test.tolist(), test_pred)
                for _ in range(iterations):
                    y_pred = self.y_predict(self.x_train.tolist(), b)
                    cost = self.compute_cost(self.y_train, y_pred)
                    costs.append(cost)
                    self.update_coefficient(learning_rate)

                    if _ % steps == 0:
                        test_pred = self.y_predict(self.x_test.tolist(), b)
                        current_accuracy = self.accuracy(self.y_test.tolist(), test_pred)
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            best_model = b
                            break

                # Show the best model
                if best_model is not None:
                    plt.title(cat)
                    color = next(self.colors)
                    self.plot_best_plot(y_pred, "OptimizedBest Fit Line", color)

                else:
                    plt.title(cat)
                    color = next(self.colors)
                    self.plot_best_plot(first_y_pred, "OptimizedBest Fit Line", color)


                # plot to verify cost function decreases
                if costs:
                    h = plt.figure('Verification')
                    plt.plot(range(iterations), costs, color=color)
                    h.show()


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


    #Isotonic regression line plotter
    def draw_isotonic_regression(self,x,y, category):
        
            #Splitting data into train and test data
            x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle= False)

            isotonic_reg = IsotonicRegression(out_of_bounds="clip")
            y_iso = isotonic_reg.fit_transform(x_train.tolist(),y_train.tolist())
            y_test_iso = isotonic_reg.transform(x_test.squeeze())
            rmse = np.sqrt(mean_squared_error(y_test,y_test_iso))

            #Not considering RMSE value as rmse doesn't seems to be a good evaluation metric for our regression
            color = next(self.colors)
            plt.scatter(x_train, y_train, marker = 'o', label='data', color= color) 
            plt.plot(x_train,y_iso,'C1-',markersize=10,label='isotonic regression' , color= color)
            plt.scatter(x_test,y_test, marker = 'x', label='data', color= color)

            x_plot = pd.Series([x_train.iloc[-1]]).append(pd.Series(x_test.tolist()))
            y_plot = pd.Series([y_iso[-1]]).append(pd.Series(y_test_iso.tolist()))
            plt.plot(x_plot,y_plot,'C1-',markersize=10,label='isotonic regression' , color= color)
            self.legendsColor.append(Line2D([0],[0],color= color, linewidth=4, label=f'{category} ({round(rmse, 2)}) ')) 


    def isotonic_plotter(self, continent =""):
        # self.df = self.master_df[self.master_df['continent'] == continent]

        #Iterating over unique categories to get frequency of each categories over a period of time 
        for cat in self.df['category'].unique():
            temp_df = self.df.copy()
            temp_df = temp_df[temp_df['category'] == cat]
            temp_df = temp_df.groupby(['year','category']).size().reset_index(name='Frequency')
            self.n = np.size(temp_df['Frequency'])
            #Limiting our dataset to consider for the data with more than 8 observations 
            if self.n > 10:
                self.draw_isotonic_regression(temp_df['year'], temp_df['Frequency'], cat)
        
        plt.legend(handles = self.legendsColor)
        plt.xlabel('Year') 
        plt.ylabel('Frequency') 
        plt.title ('Isotonic Regression of Category Frequency across Year %s'%continent) 
        plt.show()


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
