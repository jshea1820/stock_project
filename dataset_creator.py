'''

Jack Shea
S&P 500 predictor
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

 


dataset = generate_dataset('2010-01-01', '2018-05-22', 'data.csv')
