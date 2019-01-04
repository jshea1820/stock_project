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

'''dataset = quandl.get("EOD/AAPL", authtoken="yxYL3SEDrxT7cnjvshBs",
                     start_date="2000-01-01",
                     end_date="2018-01-01")
print(dataset['Close'])'''

dataset = generate_dataset('2000-01-01', '2018-01-01')
