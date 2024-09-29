import pandas as pd
#import matplotlib.pyplotas plt
import sweetviz
from AutoClean import AutoClean
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from clusteval import clusteval
import numpy as np


irco =  pd.read_csv(r"D:\project 2\Dataset\given\Project_185(coal_forecasting)\Project_185(coal_forecasting).csv")

##  Push data in sql ##

from sqlalchemy import create_engine

# Credentials to connect to Database

user = 'root'  # user name
pw = 'patilasp'  # password
db = 'pro2'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# to_sql() - function to push the dataframe onto a SQL table.

irco.to_sql('irol', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from irol;'
df = pd.read_sql_query(sql, engine)






# Exclude 'Date' column
columns_to_analyze = ['Coal_RB_4800_FOB_London_Close_USD', 
                      'Coal_RB_5500_FOB_London_Close_USD', 
                      'Coal_RB_5700_FOB_London_Close_USD', 
                      'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
                      'Coal_India_5500_CFR_London_Close_USD']

# Calculate statistics for the specified columns
stats = df[columns_to_analyze].agg(['mean', 'median', lambda x: x.mode().iloc[0], 'kurtosis', 'skew', 'min', 'max', 'std', 'var']).transpose()

# Rename the columns for better readability
stats.columns = ['Mean', 'Median', 'Mode', 'Kurtosis', 'Skewness', 'Min', 'Max', 'Standard Deviation', 'Variance']

# Display the statistics
print(stats)


# Save the statistics to a CSV file
stats.to_csv('Before_statistics.csv')

# Display the path to the saved file
print("Statistics saved to 'Before_statistics.csv'")



## Checking Missing values ##
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

## Treating missing values ##

from sklearn.impute import KNNImputer
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

## Using forward filling we fill the date ##

# Assuming 'Date' column is not datetime yet, convert it to datetime
df['Date'] = pd.to_datetime(df['Date']) 


# Impute missing values in 'Date' column using forward filling

df['Date'] = df['Date'].fillna(method='ffill')



# Check if there are any missing values left
print("Remaining missing values in 'Date' column:", df['Date'].isnull().sum())

# Display the first few rows of the DataFrame
print(df.head())


# Select columns for imputation
columns_to_impute = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD'
]

# Create KNNImputer object
imputer = KNNImputer(n_neighbors=4)

# Impute missing values
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Print the imputed dataframe
print(df)

print("Dataset Info:")
print(df.info())


print("\nSummary Statistics:")
print(df.describe())








## Time series plot ##

import matplotlib.pyplot as plt

# Selecting only numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Plotting time series for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df[column], marker='o', linestyle='-')
    plt.title(f'Time Series Plot of {column}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()



## Sesonal decomposition ##
## we do this to Data Cleaning,Forecasting,Isolating Trend,Understanding Seasonal Patterns etc ##

from statsmodels.tsa.seasonal import seasonal_decompose

# Selecting only numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Seasonal Decomposition for each numerical column
for column in numerical_columns:
    result = seasonal_decompose(df[column], model='additive', period=1)
    
    # Plotting the decomposition
    plt.figure(figsize=(14, 8))
    plt.subplot(411)
    plt.plot(df['Date'], result.observed, label='Observed')
    plt.legend()
    
    plt.subplot(412)
    plt.plot(df['Date'], result.trend, label='Trend')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(df['Date'], result.seasonal, label='Seasonal')
    plt.legend()
    
    plt.subplot(414)
    plt.plot(df['Date'], result.resid, label='Residual')
    plt.legend()
    
    plt.suptitle(f'Seasonal Decomposition of {column}', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()
    
# ACF(Auto corelation function) ##

## here we decide Lags based on 
# Short-term Patterns: If you are interested in short-term patterns or seasonality, you may want to set lags to a smaller value (e.g., 10-20).
# Long-term Patterns: For long-term patterns or trends, you may need to use a larger lags value (e.g., 50 or more). ##

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Plot autocorrelation for all columns
for column in df.columns:
    if df[column].dtype != 'object':
        plt.figure(figsize=(10, 6))
        plot_acf(df[column], lags=50, title=f'Autocorrelation Plot of {column}')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()


## PACF ##

from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# Plot partial autocorrelation for all columns
for column in df.columns:
    if df[column].dtype != 'object':
        plt.figure(figsize=(10, 6))
        plot_pacf(df[column], lags=50, title=f'Partial Autocorrelation Plot of {column}')
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')
        plt.show()


## Outliers Detection ##
import seaborn as sns
import matplotlib.pyplot as plt

# Exclude 'Date' column and any other non-numeric columns
numeric_columns = [col for col in df.columns if df[col].dtype != 'object']

# Create boxplots for each numeric column
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

## By visualizing Box plot got to know that outliers are present in our data ##


## getting count of outliers for each column ##

import numpy as np

# Define a function to count outliers based on a criterion
def count_outliers(column):
    mean = column.mean()
    std = column.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)

# Specify the column names
columns_to_check = [
    'Coal_RB_4800_FOB_London_Close_USD', 
    'Coal_RB_5500_FOB_London_Close_USD', 
    'Coal_RB_5700_FOB_London_Close_USD', 
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
    'Coal_India_5500_CFR_London_Close_USD'
]

# Create a dictionary to store outlier counts for each column
outlier_counts = {}

# Iterate over each specified column and count outliers
for column_name in columns_to_check:
    column = df[column_name]
    outlier_counts[column_name] = count_outliers(column)

# Print the outlier counts for each column
for column_name, count in outlier_counts.items():
    print(f"Column '{column_name}' has {count} outliers.")


##  outlierof each column 
'''
Column 'Coal_RB_4800_FOB_London_Close_USD' has 10 outliers.
Column 'Coal_RB_5500_FOB_London_Close_USD' has 5 outliers.
Column 'Coal_RB_5700_FOB_London_Close_USD' has 5 outliers.
Column 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD' has 5 outliers.
Column 'Coal_India_5500_CFR_London_Close_USD' has 5 outliers '''






## Using Histogram determin skewness of data ##
## As per Visualization our whole column data is right skeweed ##
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
columns_to_plot = ['Coal_RB_4800_FOB_London_Close_USD', 
                   'Coal_RB_5500_FOB_London_Close_USD', 
                   'Coal_RB_5700_FOB_London_Close_USD', 
                   'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
                   'Coal_India_5500_CFR_London_Close_USD']

for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()


# Method used for treating outlier #
## reducing the impact of outliers and improving the symmetry of the distribution, transformation may be a better choice.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Apply logarithmic transformation to the numeric columns
transformed_data = np.log1p(df[numeric_columns])

# Create boxplots of the transformed data for each column
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=transformed_data[column])
    plt.title(f'Boxplot of Log-transformed {column}')
    plt.xlabel('Transformed Values')
    plt.show()






# Exclude 'Date' column
columns_to_analyze = ['Coal_RB_4800_FOB_London_Close_USD', 
                      'Coal_RB_5500_FOB_London_Close_USD', 
                      'Coal_RB_5700_FOB_London_Close_USD', 
                      'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
                      'Coal_India_5500_CFR_London_Close_USD']

# Calculate statistics for the specified columns
stats = df[columns_to_analyze].agg(['mean', 'median', lambda x: x.mode().iloc[0], 'kurtosis', 'skew', 'min', 'max', 'std', 'var']).transpose()

# Rename the columns for better readability
stats.columns = ['Mean', 'Median', 'Mode', 'Kurtosis', 'Skewness', 'Min', 'Max', 'Standard Deviation', 'Variance']

# Display the statistics
print(stats)


# Save the statistics to a CSV file
stats.to_csv('AFTER_statistics.csv')

# Display the path to the saved file
print("Statistics saved to 'AFTER_statistics.csv")




# Set the first column (date column) as the index
df.set_index('Date', inplace=True)

## Checking Stationarity of DATA ##                                                                                                                                                                                       
from statsmodels.tsa.stattools import adfuller

# Function to perform ADF test and return True if stationary, False if non-stationary
def is_stationary(column):
    result = adfuller(column)
    return result[1] <= 0.05  # Check if p-value is less than or equal to 0.05

# Count of stationary and non-stationary columns
stationary_count = 0
non_stationary_count = 0

# Check stationarity for each column
for column in df.columns:
    if df[column].dtype != 'object':  # Check if column is numeric
        if is_stationary(df[column]):
            stationary_count += 1
        else:
            non_stationary_count += 1

# Print the counts
print(f"Number of stationary columns: {stationary_count}")
print(f"Number of non-stationary columns: {non_stationary_count}")


# Number of stationary columns: 0
# Number of non-stationary columns: 5


## Used Seasonal Decomposition for making data stationary ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# Specify the columns for seasonal decomposition
columns = ['Coal_RB_4800_FOB_London_Close_USD', 
           'Coal_RB_5500_FOB_London_Close_USD', 
           'Coal_RB_5700_FOB_London_Close_USD', 
           'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
           'Coal_India_5500_CFR_London_Close_USD']

# Perform seasonal decomposition for each column
for column in columns:
    result = seasonal_decompose(df[column], model='additive', period=12)  # Assuming a seasonal period of 12 months
    
    # Plot the original data, trend, seasonal, and residual components
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(df[column], label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(result.resid, label='Residual')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.suptitle(f'Seasonal Decomposition of {column}')
    plt.show()


## checking Stationarity of data once again ##


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Assuming 'df' is your DataFrame with the time series data
columns = ['Coal_RB_4800_FOB_London_Close_USD', 
           'Coal_RB_5500_FOB_London_Close_USD', 
           'Coal_RB_5700_FOB_London_Close_USD', 
           'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
           'Coal_India_5500_CFR_London_Close_USD']

# Initialize counters
stationary_count = 0
non_stationary_count = 0

for column in columns:
    result = seasonal_decompose(df[column], model='additive', period=12)  # Adjust period as needed
    
    # Extract the residual component
    residual = result.resid.dropna()
    
    # Perform the ADF test
    adf_result = adfuller(residual)
    
    # Perform the KPSS test
    kpss_result = kpss(residual)
    
    # Check stationarity
    if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
        stationary_count += 1
    else:
        non_stationary_count += 1

    # Print test results
    print(f'ADF p-value for {column}: {adf_result[1]}')
    print(f'KPSS p-value for {column}: {kpss_result[1]}')

print(f'Number of stationary columns: {stationary_count}')
print(f'Number of non-stationary columns: {non_stationary_count}')



## Number of stationary columns: 5
## Number of non-stationary columns: 0


from pmdarima.arima import auto_arima

# Assuming 'df' is your DataFrame with the time series data
columns = ['Coal_RB_4800_FOB_London_Close_USD', 
           'Coal_RB_5500_FOB_London_Close_USD', 
           'Coal_RB_5700_FOB_London_Close_USD', 
           'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
           'Coal_India_5500_CFR_London_Close_USD']

for column in columns:
    # Fit auto ARIMA model
    model = auto_arima(df[column], seasonal=True, m=12, suppress_warnings=True)  # Assuming monthly seasonality
    
    # Print the optimal (p, d, q) parameters
    print(f'Optimal (p, d, q) for {column}: ({model.order[0]}, {model.order[1]}, {model.order[2]})')





                                ## ARIMA MODEL ##

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from datetime import timedelta

# Specify the (p, d, q) values for each column
params = {
    'Coal_RB_4800_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_5500_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_5700_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD': (0, 1, 0),
    'Coal_India_5500_CFR_London_Close_USD': (0, 1, 0)
}

results = []

for column, (p, d, q) in params.items():
    train = df[column][:-60]  # Use all data except the last 2 months (60 days) for training
    test = df[column][-62:]   # Use the last 2 months + 2 dadys (62 days) for testing

    # Fit ARIMA model
    model = ARIMA(train, order=(p, d, q))
    fit_model = model.fit()

    # Forecast for the next 2 months (60 days)
    forecast = fit_model.forecast(steps=62)[-2:]  # Forecast only the last 2 days

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test[-2:], forecast)  # Calculate MAPE only for the last 2 days

    # Append results to the list
    results.append({
        'Column': column,
        'Model': 'ARIMA',
        'Actual': test[-2:].tolist(),  # Include only the last 2 days in the actual values
        'Predicted': forecast.tolist(),
        'MAPE': mape * 100  # Convert MAPE to percentage
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('Arima_results_N.csv', index=False)


                            ## Sarima Model ##
                            
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

# Specify the (p, d, q) values for each column
params = {
    'Coal_RB_4800_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_5500_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_5700_FOB_London_Close_USD': (0, 1, 0),
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD': (0, 1, 0),
    'Coal_India_5500_CFR_London_Close_USD': (0, 1, 0)
}

results = []

for column, (p, d, q) in params.items():
    train = df[column][:-60]  # Use all data except the last 2 months (60 days) for training
    test = df[column][-62:]   # Use the last 2 months + 2 days (62 days) for testing

    # Fit SARIMA model
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(0, 0, 0, 0))  # Assuming no seasonality for simplicity
    fit_model = model.fit()

    # Forecast for the next 2 months (60 days)
    forecast = fit_model.forecast(steps=62)[-2:]  # Forecast only the last 2 days

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test[-2:], forecast)  # Calculate MAPE only for the last 2 days

    # Append results to the list
    results.append({
        'Column': column,
        'Model': 'SARIMA',
        'Actual': test[-2:].tolist(),  # Include only the last 2 days in the actual values
        'Predicted': forecast.tolist(),
        'MAPE': mape * 100  # Convert MAPE to percentage
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('Sarima_results_N.csv', index=False)

                        
                                ## GBM Model ##


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming your DataFrame is named 'df' with the index as the date column
columns_to_forecast = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD'
]
# and the columns you want to forecast as 'columns_to_forecast'

# Splitting the data into train and test sets
train = df[:-60]  # Assuming you want to keep the last 60 days for testing
test = df[-60:]

# Creating a dictionary to store results
results = {
    'Column': [],
    'Model': [],
    'Actual Value': [],
    'Predicted Value': [],
    'MAPE (%)': []
}

for column in columns_to_forecast:
    # Extracting the target column
    y_train = train[column].values
    y_test = test[column].values
    
    # Creating the features
    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    
    # Creating and fitting the model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    # Forecasting for the next two months
    future_index = np.arange(len(train), len(train) + 60).reshape(-1, 1)
    future_forecast = model.predict(future_index)
    
    # Calculating MAPE
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test)) * 100
    
    # Appending results to the dictionary
    results['Column'].append(column)
    results['Model'].append('GBM')
    results['Actual Value'].append(y_test[:2].tolist())  # Actual values for the next two days
    results['Predicted Value'].append(future_forecast[:2].tolist())  # Forecasted values for the next two days
    results['MAPE (%)'].append(mape)

# Creating a DataFrame from the results
results_df = pd.DataFrame(results)

# Creating a CSV file
results_df.to_csv('GBM_forecast_results.csv', index=False)



                        ## XGBoost Model ##

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame with the index as the date column
columns_to_forecast = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD'
]
# and the columns you want to forecast as 'columns_to_forecast'

# Splitting the data into train and test sets
train = df[:-60]  # Assuming you want to keep the last 60 days for testing
test = df[-60:]

# Creating a dictionary to store results
results = {
    'Column': [],
    'Model': [],
    'Actual Value': [],
    'Predicted Value': [],
    'MAPE (%)': []
}

for column in columns_to_forecast:
    # Extracting the target column
    y_train = train[column].values
    y_test = test[column].values
    
    # Creating the features
    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(train) + 60).reshape(-1, 1)
    
    # Creating and fitting the model
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Forecasting for the next two months
    forecast = model.predict(X_test)
    
    # Calculating MAPE
    mape = mean_absolute_percentage_error(y_test, forecast[:60]) * 100
    
    # Appending results to the dictionary
    results['Column'].append(column)
    results['Model'].append('XGBoost')
    results['Actual Value'].append(y_test[:2].tolist())  # Actual values for the next two days
    results['Predicted Value'].append(forecast[:2].tolist())  # Forecasted values for the next two days
    results['MAPE (%)'].append(mape)

# Creating a DataFrame from the results
results_df = pd.DataFrame(results)

# Writing the results to a CSV file
results_df.to_csv('XGB_forecast_results.csv', index=False)




                                ## TBTS Model ##

'''
from tbats import TBATS
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Assuming your data is in a DataFrame called 'df' with a 'Date' column and other columns for each time series
# 'Date' column should be in datetime format and set as index
# Replace 'cols_to_forecast' with the columns you want to forecast

cols_to_forecast = ['Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD', 'Coal_RB_5700_FOB_London_Close_USD',
                     'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 'Coal_India_5500_CFR_London_Close_USD']

# Initialize an empty DataFrame to store the forecasts and evaluation metrics
forecasts = pd.DataFrame(columns=['Column', 'Model', 'Actual', 'Predicted', 'MAPE'])

for col in cols_to_forecast:
    # Fit the TBATS model
    model = TBATS(seasonal_periods=(7, 30.5))
    fitted_model = model.fit(df[col])

    # Forecast the next 60 days (change as needed)
    forecast = fitted_model.forecast(steps=60)

    # Calculate MAPE
    actual_values = df.loc[:, col].tail(2).values  # Last two actual values for training data
    predicted_values = forecast[:2]  # First two forecasted values for test data
    mape = mean_absolute_percentage_error(actual_values, predicted_values)

    # Append the forecasts to the DataFrame
    forecast_df = pd.DataFrame({'Column': [col] * 2,
                                 'Model': ['TBATS'] * 2,
                                 'Actual': [actual_values[0], actual_values[1]],
                                 'Predicted': [predicted_values[0], predicted_values[1]],
                                 'MAPE': [mape] * 2})
    forecasts = pd.concat([forecasts, forecast_df], ignore_index=True)

# Save the forecasts to a CSV file
forecasts.to_csv('tbats_forecasts.csv', index=False)
'''
## One more try for tbts model ##
from tbats import TBATS, BATS
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame with the index as the date column
columns_to_forecast = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD'
]
# and the columns you want to forecast as 'columns_to_forecast'

# Splitting the data into train and test sets
train = df[:-60]  # Assuming you want to keep the last 60 days for testing
test = df[-60:]

# Creating a dictionary to store results
results = {
    'Column': [],
    'Model': [],
    'Actual Value': [],
    'Predicted Value': [],
    'MAPE': []
}

for column in columns_to_forecast:
    # Extracting the target column
    y_train = train[column].values
    y_test = test[column].values
    
    # Creating and fitting the model
    estimator = TBATS(seasonal_periods=[7, 30.5])
    model = estimator.fit(y_train)
    
    # Forecasting for the next two months
    forecast = model.forecast(steps=60)
    
    # Calculating MAPE
    mape = mean_absolute_percentage_error(y_test, forecast[:60]) * 100
    
    # Appending results to the dictionary
    results['Column'].append(column)
    results['Model'].append('TBATS')
    results['Actual Value'].append(y_test[:2].tolist())  # Actual values for the next two days
    results['Predicted Value'].append(forecast[:2].tolist())  # Forecasted values for the next two days
    results['MAPE'].append(mape)

# Creating a DataFrame from the results
results_df = pd.DataFrame(results)

# Writing the results to a CSV file
results_df.to_csv('TBTS_forecast_results.csv', index=False)







                    ##  Hypothesis Testing  ##
        
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi

# Load MAPE scores from CSV files into DataFrames
xgb_results = pd.read_csv('XGB_forecast_results.csv')
tbts_results = pd.read_csv('TBTS_forecast_results.csv')
sarima_results = pd.read_csv('Sarima_results_N.csv')
gbm_results = pd.read_csv('GBM_forecast_results.csv')
arima_results = pd.read_csv('Arima_results_N.csv')

# Perform Friedman test
data = pd.DataFrame({
    'XGBoost': xgb_results['MAPE_xgb'],
    'TBATS': tbts_results['MAPE_tbts'],
    'SARIMA': sarima_results['MAPE_sarima'],
    'GBM': gbm_results['MAPE_gbm'],
    'ARIMA': arima_results['MAPE_arima']
})
ranked_data = data.rank(axis=1)
f_value, p_value = friedmanchisquare(*ranked_data.values.T)

# Perform Nemenyi post-hoc test if Friedman test is significant
nemenyi_results = None
if p_value < 0.05:
    models = ['XGBoost', 'TBATS', 'SARIMA', 'GBM', 'ARIMA']
    all_results = pd.concat([xgb_results['MAPE_xgb'], tbts_results['MAPE_tbts'], sarima_results['MAPE_sarima'],
                              gbm_results['MAPE_gbm'], arima_results['MAPE_arima']], keys=models, names=['Model'])
    all_results = all_results.reset_index()
    nemenyi_results = posthoc_nemenyi(all_results, val_col='value', group_col='Model')

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Model': ['XGBoost', 'TBATS', 'SARIMA', 'GBM', 'ARIMA'],
    'MAPE (%)': [xgb_results['MAPE_xgb'].mean(), tbts_results['MAPE_tbts'].mean(), sarima_results['MAPE_sarima'].mean(),
                 gbm_results['MAPE_gbm'].mean(), arima_results['MAPE_arima'].mean()],
    'Friedman p-value': p_value,
    'Nemenyi Results': [nemenyi_results] * 5
})

# Save the results to a CSV file
results_df.to_csv('model_comparison_results.csv', index=False)



















































































## LSTM ##

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Assuming 'df' is your DataFrame with 'Date' as the index and columns for each time series
# Replace 'cols_to_forecast' with the columns you want to forecast

cols_to_forecast = ['Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD', 'Coal_RB_5700_FOB_London_Close_USD',
                     'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 'Coal_India_5500_CFR_London_Close_USD']

# Initialize an empty DataFrame to store the forecasts and evaluation metrics
forecasts_lstm = pd.DataFrame(columns=['Column', 'Model', 'Actual', 'Predicted', 'MAPE'])

for col in cols_to_forecast:
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[col]])

    # Prepare the data for LSTM
    def create_dataset(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    time_steps = 60  # Adjust as needed
    X, y = create_dataset(scaled_data, time_steps)

    # Reshape data for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the LSTM model
    model.fit(X, y, epochs=100, batch_size=32)

    # Forecast the next 2 steps (change as needed)
    test_data = scaled_data[-time_steps:]
    test_data = np.reshape(test_data, (1, test_data.shape[0], 1))
    forecast = model.predict(test_data)

    # Inverse transform the forecasted data
    forecast = scaler.inverse_transform(forecast)

    # Calculate MAPE
    actual_values = df.loc[:, col].tail(2).values  # Last two actual values for training data
    predicted_values = forecast[:2]  # First two forecasted values for test data
    mape = mean_absolute_percentage_error(actual_values, predicted_values)

    # Append the forecasts to the DataFrame
    forecast_df = pd.DataFrame({'Column': [col] * 2,
                                 'Model': ['LSTM'] * 2,
                                 'Actual': actual_values,
                                 'Predicted': predicted_values.flatten(),
                                 'MAPE': [mape] * 2})
    forecasts_lstm = pd.concat([forecasts_lstm, forecast_df], ignore_index=True)

# Save the forecasts to a CSV file
forecasts_lstm.to_csv('lstm_forecasts.csv', index=False)

















