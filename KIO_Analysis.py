import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data      # Allows you to read stock info directly from the internet.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



desired_width = 450
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',15)

kio = data.DataReader("KIO.JO",
                       start='2007-1-1',
                       end='2021-12-13',
                       data_source='yahoo')

kio['Close'].plot(title='KIO Closing Price')
plt.xlabel('Date')
plt.ylabel('Share price of KIO')
plt.legend()

plt.show()

# Quick stats on the data.
print(kio.describe())

# Finding the lowest price over the past 14 years.
# Find the Date relating to the Min value.
min_price = kio['Close'].min()
print(f"The lowest price in the past 14 years is:  {min_price}")
print(f"The lowest price was recorded on {kio.index[kio['Close'] == 2535.0][0]}")

# Finding the highest price over the past 14 years.
max_price = kio['Close'].max()
# Find the Date (which in this case is the index) relating to the Max value.
print(f"The lowest price in the past 14 years is:  {max_price}")
print(f"The lowest price was recorded on {kio.index[kio['Close'] == 79079.0][0]}")

# Boxplot for the lowest to highest share prices over the past 14 years.
sns.boxplot(data=kio['Close'])
plt.show()

# Adding a column for the daily % in share price.
for n in kio['Close']:
    kio['Daily % Change'] = kio['Close'].pct_change() * 100

# Need to create further column showing whether the closing price moved up or down.
# This will make the "y" variable/Close that is being tested a classification matter.




# Splitting the data into train and test data.
# Excluding the "Close" share price from the Training Data since this is what needs to be predicted.
X = kio.drop("Close", axis=1)
y = kio["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The default estimators/no. of trees is 10 - here simply using 25. Will add more and test for overfitting later.
rfc = RandomForestClassifier(n_estimators=25)

# Fitting the training data to the model.
rfc.fit(X_train, y_train)



