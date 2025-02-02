import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

###################### Intro and Getting Stock Price Data - Python Programming for Finance p.1-2 ######################
# https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/


start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)

# df = web.DataReader('TSLA', "google", start, end)
# df2 = web.DataReader('MCO', "google", start, end)
# df.to_csv('TSLA.csv')
# df2.to_csv('MCO.csv')


df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
df2 = pd.read_csv('mco.csv', parse_dates=True, index_col=0)

plt.figure(0)
df[['Open','Close']].plot()
# plt.show()

print(df[['High','Low']].tail())
plt.figure(1)
df2[['Open','Close']].plot()
# plt.show()

###################### Basic stock data Manipulation - Python Programming for Finance p.3 ######################
# https://pythonprogramming.net/stock-data-manipulation-python-programming-for-finance/

plt.figure(3)
figTitle = 'Moving average and Volume'
plt.title(figTitle)
windowSize = 100

df['100ma'] = df['Close'].rolling(window=windowSize, min_periods=0).mean()


ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)

ax1.plot(df.index, df['Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])


###################### More stock manipulations - Python Programming for Finance p.4 ######################
# https://pythonprogramming.net/more-stock-data-manipulation-python-programming-for-finance/
figTitle = 'Resampled (decreased frequency) to 10 days'

plt.figure(4)
plt.title(figTitle)
df_ohlc = df['Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

# plt.show()
###################### Automating getting the S&P 500 list - Python Programming for Finance p.5 ######################
# https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/

import bs4 as bs
import pickle
import requests

def save_sp500_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers)

    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


# sp500 = save_sp500_tickers()



exit(1)
