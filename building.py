import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics as stat
#import xlsxwriter
import math
import numpy as np
from matplotlib import interactive
interactive(True)
import statsmodels as stm
import statsmodels.tsa.seasonal
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

try: # Normal offline run
    data = pd.read_excel(r'C:\Users\Yiannis\python\pv\Corrected_Data.xlsx',sheet_name="Data")
    buildingdata = data.values.tolist()
except: # For Colab 
    data = pd.read_excel(r'/content/drive/MyDrive/Colab Notebooks/Corrected_Data.xlsx.xlsx',sheet_name="Data")
    buildingdata = data.values.tolist()



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


    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #plt.gcf().autofmt_xdate()
    plt.title(title)
    #plt.figure()
    plt.plot(plotpvy) #Needs two arrays, cannot handle a list of lists as input
    

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

def display_ml_error_indicators(test_sety,y_1):
    mse = mean_squared_error(np.array(test_sety).reshape(-1, 1),y_1)
    print("RMSE: ", mse**(1/2.0)) # Mean Squared Error - The criterion for which Regression is done by default

    mae = mean_absolute_error(test_sety, y_1)
    print("MAE: ", mae) # Mean Absolute Error

    rscore = regr_1.score(train_setx, train_sety)
    print("R-squared:", rscore) # R-Squared

    
    #smape
    def smape(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    testy = np.array(test_sety).reshape(-1, 1)
    predy = np.array(y_1).reshape(-1, 1)
    testy[testy == 0] = 1
    predy[predy == 0] = 1
    
    smape1 = smape(testy,predy)
    print("sMAPE:",smape1)
    '''
    #mape cannot be calculated because of divide by zero error

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = mean_absolute_percentage_error(test_sety, y_1)
    print("MAPE:",mape)
    '''

bhourly = []
pvhourly = []
weatherdata = []

# Data 
hourlystatslabel = 'Hourly Statistics'
btime = wdatetotimestamp(buildingdata)

for i in range(len(buildingdata)):
    bhourly.append([btime[i],buildingdata[i][6]])
    pvhourly.append([btime[i],buildingdata[i][7]])




# Calculate statistics

# Hourly stats for building consumption
bhmeansyearly = hourlymeans(bhourly) # Hourly mean consumption for the whole year
bhstdevyearly = standarddeviation(bhourly) # Hourly deviation of consumption for the whole year

bhmeanssummer = hourlymeans(bhourly, 'Summer') # Hourly mean consumption for summer months
bhstdevsummer = standarddeviation(bhourly, 'Summer') # Hourly deviation of consumption for summer months

bhmeanswinter = hourlymeans(bhourly, 'Winter') # Hourly mean consumption for winter months
bhstdevwinter = standarddeviation(bhourly, 'Winter') # Hourly deviation of consumption for winter months

# Hourly stats for PV production
pvhmeansyearly = hourlymeans(pvhourly) # Hourly mean production for the whole year
pvhstdevyearly = standarddeviation(pvhourly) # Hourly deviation of production for the whole year

pvhmeanssummer = hourlymeans(pvhourly, 'Summer') # Hourly mean production for summer months
pvhstdevsummer = standarddeviation(pvhourly, 'Summer') # Hourly deviation of production for summer months

pvhmeanswinter = hourlymeans(pvhourly, 'Winter') # Hourly mean production for winter months
pvhstdevwinter = standarddeviation(pvhourly, 'Winter') # Hourly deviation of production for winter months

# Prepeare legend table
blegend = (hourlystatslabel,'bhmeansyearly: Hourly mean consumption for the whole year',
           'bhstdevyearly: Hourly deviation of consumption for the whole year',
           'bhmeanssummer: Hourly mean consumption for summer months',
           'bhstdevsummer: Hourly deviation of consumption for summer months',
           'bhmeanswinter: Hourly mean consumption for winter months',
           'bhstdevwinter: Hourly deviation of consumption for winter months',
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
bvalues = []
pvvalues = []
wvalues = []
wvalues1temp = []
wvalues2humid = []
wvalues3wind = []
wvalues4cloud = []
wvalues5rad = []

for i in range(len(bhourly)): # Check if dates match and prepare the lists for each value
    try: # Will raise an error when trying the time when DST starts, ignore that value -will omit a (erroneous) value of the original index
        wvalues2humid.append(buildingdata[i][9]) 
        wvalues3wind.append(buildingdata[i][10])
        wvalues4cloud.append(buildingdata[i][11])
        wvalues5rad.append(buildingdata[i][12])
           
        datesinstring.append(time.strftime('%Y-%m-%d %H:%M:%S', bhourly[i][0]))
        day.append(time.strftime('%d', bhourly[i][0]))
        month.append(time.strftime('%m', bhourly[i][0]))
        # Season
        if 3 > int(time.strftime('%m', bhourly[i][0])) >= 6:
            season.append(2) #Spring
        elif 6 > int(time.strftime('%m', bhourly[i][0])) >= 9:
            season.append(1) #Summer
        elif 9 > int(time.strftime('%m', bhourly[i][0])) >= 12:
            season.append(3) #Autumn
        else:
            season.append(4) #Winter
        year.append(time.strftime('%Y', bhourly[i][0]))
        hour.append(time.strftime('%H', bhourly[i][0]))
        weekday.append(time.strftime('%w', bhourly[i][0])) # Weekday needs to be taken into account for consumption - it is not exported to excel
        bvalues.append(bhourly[i][1]) 
        pvvalues.append(pvhourly[i][1])
            
        wvalues1temp.append(buildingdata[i][8])

    except:
        pass


plt.figure(1)
bplot(bhourly,'consumption') # Plot historical energy consumption
plt.show

plt.figure(2)
bplot(pvhourly,'production') # Plot historical energy production
plt.show

# Plot autocorrelation

plt.figure(3)
plt.title("Autocorrelation of PV energy production")
plt.acorr(pvvalues,maxlags=10)

plt.figure(4)
plt.title("Autocorrelation of energy consumption")
plt.acorr(bvalues,maxlags=10)


# Time Series Decomposition to identify trend and seasonal components
try:
    seasonalpv = stm.tsa.seasonal.seasonal_decompose(pvvalues, period=24*30) # Monthly intervals
    statsmodels.tsa.seasonal.DecomposeResult.plot(seasonalpv, observed=True, seasonal=True, trend=True, resid=True, weights=False)
except: # Normal syntax is not compatible with colab, an exception is raised and I resolve it by writing the colab comatible syntax
    seasonalpv = stm.tsa.seasonal.seasonal_decompose(pvvalues, freq=24*30) # Monthly intervals
    statsmodels.tsa.seasonal.DecomposeResult.plot(seasonalpv)


# Decision tree Regressor - Predict values
regr_1 = DecisionTreeRegressor(max_depth=10) # If max_depth is too low it won't take into account all the input features
weather_regr = []
train_setx = [] # Train set is the first 80% of the data
train_sety = []
test_setx = [] # Test set is the last 20% of the data
test_sety = []
train_time = []
test_time = []
#print(np.corrcoef(season,pvvalues)[0, 1])
for i in range(len(wvalues1temp)): # Prepare to calculate data using multiple weather variables
    if i < len(wvalues1temp)*8/10:
        train_setx.append([wvalues5rad[i],hour[i],month[i],season[i]])
        train_sety.append(pvvalues[i])
        train_time.append(datesinstring[i])
    else:
        test_setx.append([wvalues5rad[i],hour[i],month[i],season[i]])
        test_sety.append(pvvalues[i])
        test_time.append(datesinstring[i])


#regr_1.fit(np.array(wvalues5rad).reshape(-1, 1), np.array(pvvalues).reshape(-1, 1))
regr_1.fit(train_setx, train_sety)

y_1 = regr_1.predict(test_setx)

plt.figure()
plt.title("Prediction vs real values to PV production")
plt.plot(test_time, test_sety, color="darkorange", label="real values", linewidth=2)
plt.plot(test_time, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)

'''
# Comparison test single data
train_setx = [] # Train set is the first 90% of the data
train_sety = []
for i in range(int(len(wvalues5rad)*8/10)):
    train_setx.append(wvalues5rad[i])
    train_sety.append(pvvalues[i])
test_setx = [] # Test set is the last 10% of the data
test_sety = []
for i in range(len(train_setx),len(wvalues5rad)):
    test_setx.append(wvalues5rad[i])
    test_sety.append(pvvalues[i])
regr_2 = DecisionTreeRegressor(max_depth=3)
weather_regr2 = []
regr_2.fit(np.array(train_setx).reshape(-1, 1), train_sety)
X_test2 = []
for i in range(100):
    X_test2.append([i*1])
y_2 = regr_2.predict(np.array(test_setx).reshape(-1, 1))
plt.figure()
plt.title("Prediction vs real values Solar radiation to PV production")
plt.scatter(np.array(test_setx).reshape(-1, 1), np.array(test_sety).reshape(-1, 1), s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(test_setx, y_2, color="cornflowerblue", label="max_depth=2", linewidth=2)
'''

display_ml_error_indicators(test_sety,y_1)



''' # Testing the DataFrame.corr function - not needed
# Dataframe Correlation
testdataframe = pd.DataFrame({'wvalues5rad':wvalues5rad,'pvvalues':pvvalues})
tcorr = testdataframe.corr(method='pearson', min_periods=1)
print(tcorr)
plt.figure(10)
plt.matshow(testdataframe.corr())
plt.show()
'''

# Export Correlation Data

s1 = pd.Series(np.corrcoef(wvalues1temp,pvvalues)[0, 1],name='Temperature-PV Production Correlation')
s2 = pd.Series(np.corrcoef(wvalues2humid,pvvalues)[0, 1],name='Humidity-PV Production Correlation')
s3 = pd.Series(np.corrcoef(wvalues3wind,pvvalues)[0, 1],name='Wind Speed-PV Production Correlation')
s4 = pd.Series(np.corrcoef(wvalues4cloud,pvvalues)[0, 1],name='Cloud Cover-PV Production Correlation')
s5 = pd.Series(np.corrcoef(wvalues5rad,pvvalues)[0, 1],name='Solar radiation-PV Production Correlation')

s6 = pd.Series(np.corrcoef(wvalues1temp,bvalues)[0, 1],name='Temperature-Power Consumption Correlation')
s7 = pd.Series(np.corrcoef(wvalues2humid,bvalues)[0, 1],name='Humidity-Power Consumption Correlation')
s8 = pd.Series(np.corrcoef(wvalues3wind,bvalues)[0, 1],name='Wind Speed-Power Consumption Correlation')
s9 = pd.Series(np.corrcoef(wvalues4cloud,bvalues)[0, 1],name='Cloud Cover-Power Consumption Correlation')
s10 = pd.Series(np.corrcoef(wvalues5rad,bvalues)[0, 1],name='Solar radiation-Power Consumption Correlation')

exportcorrdata = pd.concat([s1,s2,s3,s4,s5,s6,s8,s9,s10], axis=1)



# Export data to Excel
exporthourlystats = pd.DataFrame({'bhmeansyearly':bhmeansyearly,'bhstdevyearly':bhstdevyearly,
                                      'bhmeanssummer':bhmeanssummer,'bhstdevsummer':bhstdevsummer,
                                      'bhmeanswinter':bhmeanswinter,'bhstdevwinter':bhstdevwinter,
                                      'pvhmeansyearly':pvhmeansyearly,'pvhstdevyearly':pvhstdevyearly,
                                      'pvhmeanssummer':pvhmeanssummer,'pvhstdevsummer':pvhstdevsummer,
                                      'pvhmeanswinter':pvhmeanswinter,'pvhstdevwinter':pvhstdevwinter
                                      }) # Hourly statistics

exportdatalegend = pd.DataFrame({'Legend':blegend}) # Legend explaining the statistics page




# Export Arranged Data

s1 = pd.Series(datesinstring, name='Date/Time')
s2 = pd.Series(day, name='Day')
s3 = pd.Series(month, name='Month')
s4 = pd.Series(year, name='Year')
s5 = pd.Series(hour, name='Hour')
s6 = pd.Series(bvalues, name='bvalues')
s7 = pd.Series(pvvalues, name='pvvalues')
s8 = pd.Series(wvalues1temp, name='air temperature [Β°C]')
s9 = pd.Series(wvalues2humid, name='relative humidity [%]')
s10 = pd.Series(wvalues3wind, name='wind speed[m/s]')
s11 = pd.Series(wvalues4cloud, name='cloudcover [%]')
s12 = pd.Series(wvalues5rad, name='global radiation [W/m^2]')

exportbdata = pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12], axis=1) # Must be written like this to allow different size lists in the same worksheet


# Create a Pandas Excel writer using XlsxWriter as the engine.
try: 
    bwriter = pd.ExcelWriter('Exported_Data.xlsx', engine='xlsxwriter')
except:
    # Colab version
    bwriter = pd.ExcelWriter('/content/ExportedData.xlsx', engine='xlsxwriter')

# Saving the Excel file
exportbdata.to_excel(bwriter, sheet_name='Data')
exportcorrdata.to_excel(bwriter, sheet_name='Correlation Data')
exporthourlystats.to_excel(bwriter, sheet_name='Hourly statistics')
exportdatalegend.to_excel(bwriter, sheet_name='Legend')

bwriter.save()




#testmissingdata(bhourly)
#testmissingdata(pvhourly)
#testmissingdata(weatherdata)
#testing
'''
test = []
for i in range(10):
    test.append(wvalues5rad[i])

print(test)
print(np.array(test).reshape(-1, 1))
'''


