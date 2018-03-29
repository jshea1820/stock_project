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

        

    def set_train_data(self, start_date, end_date):
        '''Slices the stock data into only the parts wanted for the training data'''
        
        
        

    
        




if __name__ == "__main__":
    stock = Stocknet('AMZN')
    
    
