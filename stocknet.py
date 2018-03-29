'''

Jack Shea and Luke Weller
Stock Prediction App Project
29 March, 2018

stocknet.py

- Python module for implementation of Stocknet class



'''

# Imported modules
import quandl
import pandas as pd
import tensorflow as tf
import numpy as np


# Class for gathering stock data, creating neural net for analysis
class Stocknet:
    
    def __init__(self, symbol):
        '''Simple initialization method that gets the stock's ticker'''
        '''and loads quandl data'''

        self.symbol = symbol

        # Reads in stock data
        print("Reading in data for stock {}".format(self.symbol))
        stock = quandl.get('WIKI/{}'.format(self.symbol))
        print("Stock successfully read in")

        # Sets number of days of data acquired
        self.num_days = len(stock['Open'])
        print("{} days worth of data".format(num_days))

        # Calculates daily change percentage values
        stock['Delta_p'] = (stock['Close'] - stock['Open']) / stock['Open']
        print("Calculated Daily % changes")
        

    def set_train_data(self, start_date, end_date):
        '''Slices the stock data into only the parts wanted for the training data'''

        
        
        
        

    
        




if __name__ == "__main__":
    stock = Stocknet('AMZN')
    
    
