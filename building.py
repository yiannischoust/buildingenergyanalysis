import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics as stat
import xlsxwriter
import math
import numpy as np
from matplotlib import interactive
interactive(True)
import statsmodels as stm
import statsmodels.tsa.seasonal
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import sklearn.linear_model
import os
import pickle
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

try: # For offline execution
    data = pd.read_excel(r'C:\python\CorrectedDataNew.xlsx',sheet_name="Data")
    buildingdata = data.values.tolist()
except: # For Google Colab 
    data = pd.read_excel(r'/content/drive/MyDrive/Colab Notebooks/CorrectedDataNew.xlsx',sheet_name="Data")
    buildingdata = data.values.tolist()

#Fix outliers
def fixoutliers(buildingdata):
	bq99 = []
	for i in range(len(buildingdata)):
		bq99.append(buildingdata[i][6])

	q99 = np.quantile(bq99,0.99)
	for i in range(len(buildingdata)):
		if buildingdata[i][6] > q99:
			buildingdata[i][6] = q99
	return buildingdata		

def datetotimestamp(buildingdata):# Convert date columns into timestamp
    btime = [] # List of timestamps
    btimestring = ""
    for i in range(len(buildingdata)):
        #btimestring =(str(buildingdata[i][2]) + "/" + str(buildingdata[i][1]) + "/" + str(buildingdata[i][0]) + " " + str(buildingdata[i][3]).split("+",1)[0] )
        btimestring =(str(buildingdata[i][2]) + "/" + str(buildingdata[i][1]) + "/" + str(buildingdata[i][0]) + " " + str(buildingdata[i][3]).split("+",1)[0] + "+0" + str(buildingdata[i][3]).split("+",1)[1] + ":00" )
        btime.append(datetime.datetime.strptime(btimestring,"%d/%m/%Y %H:%M:%S%z").timetuple())
    return btime

def wdatetotimestamp(buildingdata):# Convert date column into timestamp
    btime = [] # List of timestamps
    btimestring = ""
    for i in range(len(buildingdata)):
        btimestring = buildingdata[i][1] + "+01:00"
        btime.append(datetime.datetime.strptime(btimestring,"%Y-%m-%d %H:%M:%S%z").timetuple())
    return btime

def testmissingdata(bhour): #Print all values which do not follow hourly sequence - Should only print DST changes in March if all is OK
    bhournew = []
    for i in range(1,len(bhour)):
        if datetime.datetime.fromtimestamp(time.mktime(bhour[i][0])) - datetime.datetime.fromtimestamp(time.mktime(bhour[i-1][0])) != datetime.timedelta(hours=1):
            print(datetime.datetime.fromtimestamp(time.mktime(bhour[i-1][0])))
            print(datetime.datetime.fromtimestamp(time.mktime(bhour[i][0])))

       
def bplot(bhour,title="empty"): #Plot the data
    bhourly1 = []
    for i in range(len(bhour)):    
        bhourly1.append(time.mktime(bhour[i][0])) #conversion to epoch time

    plotpvx = []
    plotpvy = []
    for i in range(len(bhour)):  
        plotpvx.append(mdates.date2num(datetime.datetime.utcfromtimestamp(bhourly1[i])))# Convert timestamp to a format matplotlib can handle
        plotpvy.append(bhour[i][1])
    plt.title(title)
    plt.plot(plotpvy)
    

def hourlymeans(bhourly, season='Yearly'): # Calculate hourly means
    hmeans = []
    hcount = []
    hourly = []

    for j in range(24):
        hmeans.append(0)
        hcount.append(0)
        for i in range(len(bhourly)):
            if season == "Yearly":
                if bhourly[i][0].tm_hour==j:
                   hmeans[j] += bhourly[i][1]
                   hcount[j] += 1
            elif season == "Summer":
                if bhourly[i][0].tm_hour==j and (bhourly[i][0].tm_mon > 3 and bhourly[i][0].tm_mon < 10):
                   hmeans[j] += bhourly[i][1]
                   hcount[j] += 1
            elif season == "Winter":
                if bhourly[i][0].tm_hour==j and (bhourly[i][0].tm_mon <= 3 or bhourly[i][0].tm_mon >= 10):
                   hmeans[j] += bhourly[i][1]
                   hcount[j] += 1
            else:
                print('Error: Season not defined')
    for j in range(24):
        if hcount[j] != 0:
            hmeans[j] = hmeans[j]/hcount[j] # Divide the sum of elements by the number of elements in each
        else:
            hmeans[j] = 0
    return hmeans

def standarddeviation(bhourly, season='Yearly'): # Calculate hourly standard deviation
    listbyhour = []
    std = []
    for j in range(24):
        listbyhour.append([])
        for i in range(len(bhourly)):
            if season == "Yearly":
                if bhourly[i][0].tm_hour==j:
                    listbyhour[j].append(bhourly[i][1])
            elif season == "Summer":
                if bhourly[i][0].tm_hour==j and (bhourly[i][0].tm_mon > 3 and bhourly[i][0].tm_mon < 10):
                    listbyhour[j].append(bhourly[i][1])
            elif season == "Winter":
                if bhourly[i][0].tm_hour==j and (bhourly[i][0].tm_mon <= 3 or bhourly[i][0].tm_mon >= 10):
                    listbyhour[j].append(bhourly[i][1])
            else:
                print('Error: Season not defined')

    for j in range(24):
        if len(listbyhour[j]) > 1:
            std.append(stat.stdev(listbyhour[j]))
        else:
            std.append(0)

    return std

def display_ml_error_indicators(test_sety,predicted_y):
    print("RMSE", format(mean_squared_error(predicted_y, test_sety, squared=False), ".0f"))    
    print("MAE", format(mean_absolute_error(predicted_y, test_sety), ".0f"))
    print("r_square", format(r2_score(predicted_y, test_sety), ".3f"))
   

#Trim production values over 99% of the rest of the data - not used
#buildingdata = fixoutliers(buildingdata)

pvhourly = []
weatherdata = []
btime = wdatetotimestamp(buildingdata)

for i in range(len(buildingdata)): # Create a list containting the timestamp and production for each hour
    pvhourly.append([btime[i],buildingdata[i][6]])

# Hourly statistics for PV production
hourlystatslabel = 'Hourly Statistics'
pvhmeansyearly = hourlymeans(pvhourly) # Hourly mean production for the whole year
pvhstdevyearly = standarddeviation(pvhourly) # Hourly deviation of production for the whole year

pvhmeanssummer = hourlymeans(pvhourly, 'Summer') # Hourly mean production for summer months
pvhstdevsummer = standarddeviation(pvhourly, 'Summer') # Hourly deviation of production for summer months

pvhmeanswinter = hourlymeans(pvhourly, 'Winter') # Hourly mean production for winter months
pvhstdevwinter = standarddeviation(pvhourly, 'Winter') # Hourly deviation of production for winter months

# Prepeare legend table
blegend = (hourlystatslabel,
           'pvhmeansyearly: Hourly mean production for the whole year',
           'pvhstdevyearly: Hourly deviation of production for the whole year',
           'pvhmeanssummer: Hourly mean production for summer months',
           'pvhstdevsummer: Hourly deviation of production for summer months',
           'pvhmeanswinter: Hourly mean production for winter months',
           'pvhstdevwinter: Hourly deviation of production for winter months')



datesinstring = []
day = []
month = []
season = []
year = []
hour = []
weekday = []
pvvalues = []
wvalues = []
wvalues1temp = []
wvalues2humid = []
wvalues3wind = []
wvalues4cloud = []
wvalues5rad = []
wvalues5rad_previoushour = []
wvalues5rad_preprevioushour = []
wvalues5rad_3hoursbefore = []
wvalues5rad_4hoursbefore = []
wvalues5rad_5hoursbefore = []
wvalues5rad_6hoursbefore = []
for i in range(len(pvhourly)): # Check if dates match and prepare the lists for each value
    try:
        wvalues2humid.append(buildingdata[i][8]) 
        wvalues3wind.append(buildingdata[i][9])
        wvalues4cloud.append(buildingdata[i][10])
        wvalues5rad.append(buildingdata[i][11])
        if i > 6:
            wvalues5rad_previoushour.append(wvalues5rad[i-1])
            wvalues5rad_preprevioushour.append(wvalues5rad[i-2])
            wvalues5rad_3hoursbefore.append(wvalues5rad[i-3])
            wvalues5rad_4hoursbefore.append(wvalues5rad[i-4])
            wvalues5rad_5hoursbefore.append(wvalues5rad[i-5])
            wvalues5rad_6hoursbefore.append(wvalues5rad[i-6])
        else:
            wvalues5rad_previoushour.append(wvalues5rad[i])
            wvalues5rad_preprevioushour.append(wvalues5rad[i])
            wvalues5rad_3hoursbefore.append(wvalues5rad[i])
            wvalues5rad_4hoursbefore.append(wvalues5rad[i])
            wvalues5rad_5hoursbefore.append(wvalues5rad[i])
            wvalues5rad_6hoursbefore.append(wvalues5rad[i])
        datesinstring.append(time.strftime('%Y-%m-%d %H:%M:%S', pvhourly[i][0]))
        day.append(time.strftime('%d', pvhourly[i][0]))
        month.append(time.strftime('%m', pvhourly[i][0]))
        
        currentmonth = int(time.strftime('%m', pvhourly[i][0]))
        # Season
        if 3 < currentmonth and currentmonth <= 6:
            season.append('Spring') #Spring
        elif 6 < currentmonth and currentmonth <= 9:
            season.append('Summer') #Summer
        elif 9 < currentmonth and currentmonth <= 12:
            season.append('Autumn') #Autumn
        elif 0 < currentmonth and currentmonth <= 3:
            season.append('Winter') #Winter
        else:
            print("error")
            
        year.append(time.strftime('%Y', pvhourly[i][0]))
        hour.append(time.strftime('%H', pvhourly[i][0]))
        weekday.append(time.strftime('%w', pvhourly[i][0]))
        pvvalues.append(pvhourly[i][1])
            
        wvalues1temp.append(buildingdata[i][7])

    except:
        print('error')

encmonth = OneHotEncoder(categories='auto',handle_unknown='ignore')
enchour = OneHotEncoder(categories='auto',handle_unknown='ignore')
encseason = OneHotEncoder(categories='auto',handle_unknown='ignore')

month_cat = encmonth.fit_transform(np.array(month).reshape(-1, 1))
month_cat = month_cat.toarray()
hour_cat = enchour.fit_transform(np.array(hour).reshape(-1, 1))
hour_cat = hour_cat.toarray()
season_cat = encseason.fit_transform(np.array(season).reshape(-1, 1))
season_cat = season_cat.toarray()

s1 = pd.Series(datesinstring, name='Date/Time')
s2 = pd.Series(day, name='Day')
s3 = pd.Series(month, name='Month')
s4 = pd.Series(year, name='Year')
s5 = pd.Series(hour, name='Hour')
s7 = pd.Series(pvvalues, name='pvvalues')
s8 = pd.Series(wvalues1temp, name='air temperature [Β°C]')
s9 = pd.Series(wvalues2humid, name='relative humidity [%]')
s10 = pd.Series(wvalues3wind, name='wind speed[m/s]')
s11 = pd.Series(wvalues4cloud, name='cloudcover [%]')
s12 = pd.Series(wvalues5rad, name='global radiation [W/m^2]')
s12_1 = pd.Series(wvalues5rad_previoushour, name='rvalues2')
s12_2 = pd.Series(wvalues5rad_preprevioushour, name='rvalues3')
s12_3 = pd.Series(wvalues5rad_3hoursbefore, name='rvalues4')
s13 = pd.Series(season, name='Season')
#Different input sets of numerical features - the sets not used in the current test are commented out
'''df = pd.DataFrame({
                    'rvalues2':wvalues5rad_previoushour,'rvalues3':wvalues5rad_preprevioushour,
                    'rvalues4':wvalues5rad_3hoursbefore,'h':hour,'globalradiation':wvalues5rad
                    }
                )'''
df = pd.DataFrame({
                    'rvalues2':wvalues5rad_previoushour,'rvalues3':wvalues5rad_preprevioushour,
                    'rvalues4':wvalues5rad_3hoursbefore,'rvalues5':wvalues5rad_4hoursbefore,'globalradiation':wvalues5rad
                    }
                )
'''df = pd.DataFrame({
                    'rvalues2':wvalues5rad_previoushour,'rvalues3':wvalues5rad_preprevioushour,
                    'rvalues4':wvalues5rad_3hoursbefore,'globalradiation':wvalues5rad
                    }
                )'''
'''df = pd.DataFrame({
                    'rvalues2':wvalues5rad_previoushour,'rvalues3':wvalues5rad_preprevioushour,
                    'rvalues4':wvalues5rad_3hoursbefore,'globalradiation':wvalues5rad,
                    'rvalues5':wvalues5rad_4hoursbefore,'rvalues6':wvalues5rad_5hoursbefore
                    }
                )'''

# Categorical input features - the variables not used in the current test are commented out
#df[encmonth.categories_[0]]= month_cat # Use month as a categorical feature
#df[enchour.categories_[0]]= hour_cat # Use hour as a categorical feature
#df[encseason.categories_[0]]= season_cat # Use season as a categorical feature

df2 = pd.DataFrame(pvvalues)
df_time = pd.DataFrame(datesinstring)

test_set_index = int(len(wvalues1temp)*0.5) # The size of the test set - set to 50% of the dataset

train_setx = df.iloc[:len(wvalues1temp)-test_set_index,:]
test_setx = df.iloc[len(wvalues1temp)-test_set_index:,:]
train_sety = df2.iloc[:len(wvalues1temp)-test_set_index,:]
test_sety = df2.iloc[len(wvalues1temp)-test_set_index:,:]
train_time = df_time.iloc[:len(wvalues1temp)-test_set_index,:]
test_time = df_time.iloc[len(wvalues1temp)-test_set_index:,:]

train_setx= train_setx.to_numpy()
test_setx = test_setx.to_numpy()

#Plots for statistics - not used for machine learning tests
plt.figure(2)
bplot(pvhourly,'production') # Plot historical energy production
plt.show


# Plot autocorrelation

plt.figure(3)
plt.title("Autocorrelation of PV energy production")
plt.acorr(pvvalues,maxlags=10)


# Time Series Decomposition to identify trend and seasonal components
try:
    seasonalpv = stm.tsa.seasonal.seasonal_decompose(pvvalues, period=24*30) # Monthly intervals
    statsmodels.tsa.seasonal.DecomposeResult.plot(seasonalpv, observed=True, seasonal=True, trend=True, resid=True, weights=False)
except: # Normal syntax is not compatible with colab, an exception is raised and I resolve it by writing the colab comatible syntax
    seasonalpv = stm.tsa.seasonal.seasonal_decompose(pvvalues, freq=24*30) # Monthly intervals
    statsmodels.tsa.seasonal.DecomposeResult.plot(seasonalpv)


weather_regr = []

# Prepare the lists of the output data and timestamps for machine learning
train_sety = []
test_sety = []
train_time = []
test_time = []
for i in range(len(wvalues1temp)): 
    if i < len(wvalues1temp)-test_set_index:
        train_sety.append(pvvalues[i])
        train_time.append(datesinstring[i])
    else:
        test_sety.append(pvvalues[i])
        test_time.append(datesinstring[i])

sc_x = preprocessing.StandardScaler()
train_setx_norm = sc_x.fit_transform(train_setx) #Normalized input data
test_setx_norm = sc_x.fit_transform(test_setx)

# Decision tree Regressor - Predict values
#regr_1 = DecisionTreeRegressor(criterion='squared_error', random_state = 0, max_depth = 22, min_samples_split=4, min_samples_leaf=20) # If max_depth is too low it won't take into account all the input features
regr_1 = make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='squared_error', random_state = 0, max_depth = 15, min_samples_split=2, min_samples_leaf=21,splitter='random'))

regr_1.fit(train_setx, train_sety)

y_1 = regr_1.predict(test_setx)


plt.figure()
plt.title("Decision Tree: Prediction vs real values to PV production")
plt.plot(test_time, test_sety, color="darkorange", label="real values", linewidth=2)
plt.plot(test_time, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)

print("Decision Tree Method:")
display_ml_error_indicators(test_sety,y_1)
rscore = regr_1.score(test_setx, test_sety)
print("R-squared:", format(rscore, ".3f")) # R-Squared

# SVM - Predict values

regr_2 = make_pipeline(preprocessing.StandardScaler(), svm.SVR(kernel='rbf',C=220000, epsilon=0.2,gamma='auto'))
#regr_2 = make_pipeline(preprocessing.StandardScaler(), svm.SVR(kernel='linear',C=30, epsilon=0.2,gamma='auto'))

regr_2.fit(train_setx, train_sety)
y_2 =  regr_2.predict(test_setx)

print("SVR:")
display_ml_error_indicators(test_sety,y_2)
rscore = regr_2.score(test_setx, test_sety)
print("R-squared:", format(rscore, ".3f")) # R-Squared


plt.figure()
plt.title("SVR: Prediction vs real values to PV production")
plt.plot(test_time, test_sety, color="darkorange", label="real values", linewidth=2)
plt.plot(test_time, y_2, color="cornflowerblue", label="predicted", linewidth=2)

# Linear Regression - Predict values

regr_3 = sklearn.linear_model.LinearRegression().fit(train_setx_norm, train_sety)
y_3 =  regr_3.predict(test_setx_norm)


print("Linear Regression:")
display_ml_error_indicators(test_sety,y_3)
test_setx_norm = np.array(test_setx_norm, dtype=float) # Need to explicitly convert to numeric or the score function gives a warning
test_sety = np.array(test_sety, dtype=float)
rscore = regr_3.score(test_setx_norm, test_sety)
print("R-squared:", format(rscore, ".3f")) # R-Squared


plt.figure()
plt.title("Linear Regression: Prediction vs real values to PV production")
plt.plot(test_time, test_sety, color="darkorange", label="real values", linewidth=2)
plt.plot(test_time, y_3, color="cornflowerblue", label="predicted", linewidth=2)


# Ensemble of Linear regression and SVR
print("Ensemble of Linear regression and SVR:")
weights = [0.5, 0.5]
models = []
models.append(('r1',regr_2))
models.append(('r2',regr_3))
ensemble = VotingRegressor(estimators=models, weights=weights)
ensemble.fit(train_setx, train_sety)

y_4 = ensemble.predict(test_setx)
display_ml_error_indicators(test_sety,y_4)

print("R-squared:", format(ensemble.score(test_setx_norm, test_sety), ".3f"))


print("Ensemble of Decision tree and SVR:")
#weights = [0.5, 0.5]
weights = [0.75, 0.25]
models = []
models.append(('r1',regr_2))
models.append(('r2',regr_1))
ensemble = VotingRegressor(estimators=models, weights=weights)
ensemble.fit(train_setx, train_sety)

y_5 = ensemble.predict(test_setx)
display_ml_error_indicators(test_sety,y_5)

print("R-squared:", format(ensemble.score(test_setx_norm, test_sety), ".3f"))


plt.figure()
plt.title("Ensemble: Prediction vs real values to PV production")
plt.plot(test_time, test_sety, color="darkorange", label="real values", linewidth=2)
plt.plot(test_time, y_5, color="cornflowerblue", label="predicted", linewidth=2)


#Grid search

# SVR Hyperparameter Tests
param_grid_svr = {'svr__C': [210000,220000,230000,240000],  
              'svr__epsilon': [0.2], 
              'svr__gamma':['auto'],
              'svr__kernel': ['rbf']}  

grid_svr = Pipeline(steps=[('scaler',preprocessing.StandardScaler()),('svr', svm.SVR())])			  

#Decision Tree Regressor hyperparameter tests
param_grid_tree = {'tree__criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                   'tree__splitter':['best', 'random'],
                   'tree__max_depth': [ 5, 10, 20, 22, 40, 50 ],
                   'tree__min_samples_split': [ 2, 4, 8 ],
                   'tree__min_samples_leaf': [1, 10, 20, 40],
                                  
                   }
param_grid_tree1 = {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                   'splitter':['best', 'random'],
                   'max_depth': [ 5, 10, 20, 22, 40, 50 ],
                   'min_samples_split': [ 2, 4, 8 ],
                   'min_samples_leaf': [1, 10, 20, 40],
                                  
                   }
param_grid_tree2 = {'tree__criterion':['squared_error'],
                   'tree__splitter':['best'],
                    'tree__random_state':[0],
                   'tree__max_depth': [ 8, 9, 10,11,12,13, 22, 40, 80, 120 ],
                   'tree__min_samples_split': [ 4 ],
                   'tree__min_samples_leaf': [20]
                                  
                   }
param_grid_tree3 = {'random_state':[0],
                    'criterion':['squared_error', 'friedman_mse'],
                   'splitter':['best', 'random'],
                   'max_depth': [ 9 ],
                   'min_samples_split': [ 2 ],
                   'min_samples_leaf': [21],
                                  
                   }
extra_tree_features = {
                       'tree__max_features':['None','auto', 'sqrt', 'log2'],
                   #'tree__min_impurity_decrease':[0,0.1,0.2,1],
                   #'tree__ccp_alpha':[0,0.1,0.5,1]
                       }
grid_params_voting = { #Tests for Ensemble Models
               'weights': [(0.625, 0.375),(0.75, 0.25),(0.875, 0.125),(1, 0)]}
 
grid_tree = DecisionTreeRegressor()		  
#grid_tree = Pipeline(steps=[('scaler',preprocessing.StandardScaler()),('tree', DecisionTreeRegressor())])	

#The selection of the above parameters can be typed here to perform the corresponding test
grid = GridSearchCV(VotingRegressor(estimators=[('regr2', regr_2),('regr1', regr_1)]), param_grid=grid_params_voting, refit = True, verbose = 3,n_jobs=-1) 
# Fitting the model for grid search 
grid.fit(train_setx, train_sety)
 
# Print best parameter after tuning 
print(grid.best_params_) 
grid_predictions = grid.predict(test_setx) 
   
# print the Score of the best parameters
print("R-squared:", format(grid.best_score_, ".3f"))

#End of Gridsearch



# Prediction 1 Model Saving 
filename = 'adegapalmela_temp_solar.sav'
pickle.dump(regr_1, open(filename, 'wb'))


# Export Correlation Data

s1 = pd.Series(np.corrcoef(wvalues1temp,pvvalues)[0, 1],name='Temperature-PV Production Correlation')
s2 = pd.Series(np.corrcoef(wvalues2humid,pvvalues)[0, 1],name='Humidity-PV Production Correlation')
s3 = pd.Series(np.corrcoef(wvalues3wind,pvvalues)[0, 1],name='Wind Speed-PV Production Correlation')
s4 = pd.Series(np.corrcoef(wvalues4cloud,pvvalues)[0, 1],name='Cloud Cover-PV Production Correlation')
s5 = pd.Series(np.corrcoef(wvalues5rad,pvvalues)[0, 1],name='Solar radiation-PV Production Correlation')

exportcorrdata = pd.concat([s1,s2,s3,s4,s5], axis=1)



# Export data to Excel
exporthourlystats = pd.DataFrame({
                                      'pvhmeansyearly':pvhmeansyearly,'pvhstdevyearly':pvhstdevyearly,
                                      'pvhmeanssummer':pvhmeanssummer,'pvhstdevsummer':pvhstdevsummer,
                                      'pvhmeanswinter':pvhmeanswinter,'pvhstdevwinter':pvhstdevwinter
                                      }) # Hourly statistics

exportdatalegend = pd.DataFrame({'Legend':blegend}) # Legend explaining the statistics page




# Export Arranged Data


exportbdata = pd.concat([s1,s2,s3,s4,s5,s7,s8,s9,s10,s11,s12,s13], axis=1) # Must be written like this to allow different size lists in the same worksheet


# Create a Pandas Excel writer using XlsxWriter as the engine.
try: 
    bwriter = pd.ExcelWriter('Exported_Data.xlsx', engine='xlsxwriter')
except:
    # Colab version
    bwriter = pd.ExcelWriter('/content/drive/MyDrive/Colab Notebooks/ExportedData.xlsx', engine='xlsxwriter')

# Saving the Excel file
exportbdata.to_excel(bwriter, sheet_name='Data')


exportcorrdata.to_excel(bwriter, sheet_name='Correlation Data')
exporthourlystats.to_excel(bwriter, sheet_name='Hourly statistics')
exportdatalegend.to_excel(bwriter, sheet_name='Legend')


bwriter.save()


