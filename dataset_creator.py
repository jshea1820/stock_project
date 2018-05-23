'''

Jack Shea
14 May, 2018

dataset_creator.py

- Creates a pandas dataframe with S&P 500 constituent company data
- Each column is the d_percentage of stock price over a given time frame


'''

# Imported modules
import quandl
import pandas as pd
import tensorflow as tf
import numpy as np
import math


def generate_dataset(start, end, filename):
    ''' Takes in a start and end date and returns a creates a csv with all the stock d_percent data'''

    # Gets list of symbols
    symbol_file = open("symbols.txt")
    symbols = symbol_file.read().split('\n')[1:]
    symbols = list(map(lambda s: s.split(',')[0].replace('.','_'), symbols))

    # Gets stock data
    all_data = pd.DataFrame()
    for index, symbol in enumerate(symbols):
        print("Getting data for {}".format(symbol))
        
        try:
            stock_data = quandl.get("WIKI/{}".format(symbol), authtoken="yxYL3SEDrxT7cnjvshBs", start_date = start, end_date = end)
        except:
            print("Failure to retrieve data for {}".format(symbol))
            continue

            
        d_percent = (stock_data['Close'] - stock_data['Open']) / stock_data['Open']
        all_data[symbol] = d_percent

    
    all_data.to_csv(filename, sep=',')


def add_s_and_p():
    ''' Adds the S&P index data to the data.csv file '''

    # Gets s&p data from file
    s_and_p = pd.read_csv('S&P 500 Historical Data_update.csv')

    # Gets data.csv file
    all_data = pd.read_csv('data.csv', index_col = 'Date')
    all_data['S&P'] = np.nan

    for i in range(s_and_p.shape[0]):
        
        # Gets price change corresponding to date
        try:
            d_percent = (float(s_and_p['Open'][i-1]) - float(s_and_p['Price'][i])) / float(s_and_p['Price'][i])
        except KeyError:
            print("Failure at index {}".format(i))
            continue

        date = s_and_p['Date'][i]
        new_date = date_format(date)
        try:
            x = all_data['S&P'][new_date]
        except KeyError:
            continue
            
        all_data['S&P'][new_date] = d_percent

    all_data = cleanup(all_data)
        
    all_data.to_csv('data_update.csv', sep=',')


def date_format(date):
    ''' Formats a given date appropriately and returns the edited string '''

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    year = date.split(' ')[2]
    month_abbr = date.split(' ')[0]
    for i in range(12):
        if months[i] == month_abbr:
            month = i+1
            break
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)
    day = date.split(' ')[1].replace(',','')
    return year + '-' + month + '-' + day


def cleanup(all_data):
    ''' Removes null observations from dataframe '''

    for date, row in all_data.iterrows():
        if math.isnan(all_data["S&P"][date]):
            print("No data for date {}".format(date))
            all_data.drop([date], inplace = True)

    for key in all_data:
        for date, row in all_data.iterrows():
            if math.isnan(all_data[key][date]):
                all_data[key][date] = 0

    return all_data




#dataset = generate_dataset('2010-01-01', '2018-05-22', 'data.csv')
add_s_and_p()






