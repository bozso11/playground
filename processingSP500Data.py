import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests


###################### Getting all company pricing data in the S&P 500 - Python Programming for Finance p.6 ######################
# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def get_data_from_google(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):

            print('Getting ticker {}'.format(ticker))
            try:
                df = web.DataReader(ticker, "google", start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except:
                print("No data was retrieved for {}".format(ticker))

        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        except:
            print("No data is available for {}".format(ticker))
        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


# compile_data()

###################### Creating massive S&P 500 company correlation table for Relationships - Python Programming for Finance p.8 ######################
# https://pythonprogramming.net/stock-price-correlation-table-python-programming-for-finance/

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')

    df_corr = df.corr()
    print(df_corr.head())

    data1 = df_corr.values

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    # plt.savefig("correlations.png", dpi = (300))
    plt.show()
###################### Preprocessing data to prepare for Machine Learning with stock data - Python Programming for Finance p.9 ######################
# https://pythonprogramming.net/preprocessing-for-machine-learning-python-programming-for-finance

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

###################### Creating targets for machine learning labels - Python Programming for Finance p.10 and 11 ######################
# https://pythonprogramming.net/targets-for-machine-learning-labels-python-programming-for-finance/
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0



df = pd.read_csv('sp500_joined_closes.csv')
print(df.head())



######################  ######################
#





######################  ######################
#


######################  ######################
#

######################  ######################
#

exit(1)
