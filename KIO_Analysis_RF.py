# Creating a model using Random Forests from sklearn.

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

# Finding the correlation between the volume and the Close price.
print(kio.corr()[['Close', 'Volume']])


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
X = kio.drop('Move Category', axis=1)
y = kio['Move Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# The default estimators/no. of trees is 10 - here simply using 25. Will add more and test for overfitting later.
rfc = RandomForestClassifier(n_estimators=25)

# Fitting the training data to the model.
rfc.fit(X_train, y_train)

# Predicting the category of price movement the next day's price will be in.
prediction = rfc.predict(X_test)

# Comparing the accuracy of the model to actual data.
print('\n Results of the Confusion Matrix')
print(confusion_matrix(y_test, prediction))
print('\n Results of the Classification Report')
print(classification_report(y_test, prediction))

rfc.score(X_test, y_test)



