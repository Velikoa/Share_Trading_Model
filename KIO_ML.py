# Creating a model using Tensorflow and Keras.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data      # Allows you to read stock info directly from the internet.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',15)

kio = data.DataReader("KIO.JO",
                       start='2007-1-1',
                       end='2021-12-13',
                       data_source='yahoo')

# Quick stats on the data.
print(kio.describe())

# Adding a column for the daily % in share price.
for n in kio['Close']:
    kio['Daily % Change'] = kio['Close'].pct_change() * 100

# Need to create further column showing whether the closing price moved up or down.
# This will make the "y" variable/Close that is being tested a classification matter.
# The movement will be broken down into 6 categories - 1 - between 0 and +/-1, 2 - between +/-1 and +/-2, etc. etc.
# The final category is when the price movement is greater than +/-5%.
kio['Move Category'] = pd.cut(x=kio['Daily % Change'], bins=[-np.inf, -5, -4, -3, -2,
                                                             -1, 0, 1, 2, 3, 4, 5, np.inf], labels=['Less than -5%',
                                                                            '-4% to -5%', '-3% to -4%',
                                                                            '-2% to -3%', '-1% to -2%',
                                                                            '0% to -1%', '0% to 1%',
                                                                            '1% to 2%', '2% to 3%',
                                                                            '3% to 4%', '4% to 5%',
                                                                            'Greater than 5%'])

# Confirm there are no missing values in the data.
# The below shows there is 1 instance on the Daily % Move and the Move Category columns.
kio.isnull().sum()
print(kio[kio['Daily % Change'].isnull() == True])

# Now going to replace the 'NaN' value with the mean/average of the entire column.
kio['Daily % Change'].fillna(value=kio['Daily % Change'].mean(), inplace=True)
kio['Move Category'].fillna(value='0% to 1%', inplace=True)
kio.head()

# Splitting the data into train and test data.
# Excluding the "Move Category" share price from the Training Data since this is what needs to be predicted.
# Need to also exclude the 'Move Category' since will be using numerical figures in the prediction process.
# Therefore, cannot be using numerical figures to predict a categorical value.
X = kio.drop(['Move Category', 'Daily % Change'], axis=1).values
y = kio['Daily % Change'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Need to scale the data for it to be easier for the model to work with the numbers.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Now transforming the test data as well.
X_test = scaler.transform(X_test)

# Creating the model.
# Determine the number of neurons to use - namely, the number of columns there are within the df.
print(X_train.shape)

model = Sequential()
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))

# Only one neuron in the last layer since only predicting one value.
model.add(Dense(1, activation='relu'))

# Compile the model.
# Using a mean-squared-error as the loss metric since this is a regression problem.
model.compile(optimizer='adam', loss='mse')

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=20, epochs=200)

# Saving the model in order to not have to re-run the entire process.
model.save('KIO_ML.h5')

# Graphing the losses of the training and test sets.
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# Predict what the 'Daily % Change' will be for the next day.


