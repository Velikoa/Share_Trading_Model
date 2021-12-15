import requests
import json
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# pd.set_option('display.max_columns', None)

api_key = 'J6YWRQKVLXABS6Y0'

# url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=J6YWRQKVLXABS6Y0'
# r = requests.get(url=url)
# data = r.json()
# data_table = data.items()
#
# print(data)
# print(data_table)

# Using the pandas format already gives you the data in dataframe format.
ts = TimeSeries(key=api_key, output_format='pandas')
fd = FundamentalData(key=api_key, output_format='pandas')

data = ts.get_daily('AAPL')
data_fund = fd.get_income_statement_annual('MSFT')

print(tabulate(data[0], headers='keys', tablefmt='psql'))
data[0]['4. close'].plot()

plt.show()

# The .T transposes the data so the dates are on the columns instead of the rows.
print(tabulate(data_fund[0].T, headers='keys', tablefmt='psql'))

comp_overview = fd.get_company_overview(symbol='MSFT')
print(tabulate(comp_overview))



