'''

Jack Shea
May 23, 2018
date_changer.py

Goes through the S&P_historical data and updates dates to correspond
with month/day/year format

'''


import pandas as pd

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

spreadsheet = pd.read_csv('S&P 500 Historical Data.csv')
for index, date in enumerate(spreadsheet['Date']):
    
    # Gets month
    new_date = ''
    date_split = date.split(' ')
    month = date_split[0]
    for i in range(12):
        if month == months[i]:
            new_date += str(i+1)
    new_date += '/'

    # Gets day
    day = date_split[1].replace(',','')
    new_date += str(int(day))
    new_date += '/'

    # Gets year
    year = date_split[2]
    new_year = year[-2:]
    new_date += new_year

    spreadsheet['Date'][index] = new_date


spreadsheet.to_csv('S&P 500 Historical Data_update.csv', sep=',')



    
    
    
    

    
            
    
    
