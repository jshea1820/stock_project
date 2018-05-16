'''

Jack Shea
S&P 500 predictor
14 May, 2018

500_stock_predictor.py

- Uses tensorflow to create a network for predicting
  S&P 500 prices


'''

# Imported modules
import quandl
import pandas as pd
import tensorflow as tf
import numpy as np


def generate_dataset(start, end):
    ''' Takes in a start and end date and returns a dataframe with all S&P 500 close
    prices in that range '''

    # Gets list of symbols
    symbol_file = open("symbols.txt")
    symbols = symbol_file.read().split(', ')
    symbols = list(map(lambda s: s.replace("'", "").replace("\n", ""), symbols))

    # Gets S&P data
    s_and_p_data = quandl.get("EOD/SPY",
                              authtoken="yxYL3SEDrxT7cnjvshBs",
                              start_date = start,
                              end_date = end)
    print(s_and_p_data)

 

    



dataset = generate_dataset('2000-01-01', '2018-01-01')
