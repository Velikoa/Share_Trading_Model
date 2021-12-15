# Due to lack of available information for historical iron ore prices and API's requiring paid subscriptions, I am
# forced to scrape the data available from indexmundi.com
# I have only been able to find historical iron ore prices from January 2017 until now but it is instead monthly prices
# and not daily ones such as the share prices for KIO.
# I will be averaging out the KIO share prices for each month in order to determine the correlation between them.

from bs4 import BeautifulSoup as soup
import requests

# Creating the user-agent relationship before parsing the html with bs4 and using it within the dataframes later.
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36"}

url = "https://www.indexmundi.com/commodities/?commodity=iron-ore&months=180"

# Retrieving information from the website.
r = requests.get(url=url, headers=headers)

# Parsing the information from the website using bs4.
html_soup = soup(r.text, 'html.parser')

# Search for the specific part within the webpage where the historical iron ore prices are within the table format.
# This, however, shows the table portion of the parsed webpage but together with all the html jargon/code as well but
# in string/text format.
price_table = html_soup.find_all("table")

# Loop through all the parsed information and only print the text.
for i in price_table:
    print(i.text)



