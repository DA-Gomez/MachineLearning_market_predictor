import yfinance as yf #yahoo finance api
import model as model
import matplotlib.pyplot as plt 
import pandas as pd

def analyze(stock): 
    stock = yf.Ticker(stock) #ticker class (downloads price history for the GSPC symbol)
    stock = stock.history(period="max") 
    # queries historical prices (each row is the price for each trading day. Date, Open, High, Low, close, Volume, dividents, stocks splits)

    del stock["Dividends"]
    del stock["Stock Splits"]


    stock = stock.loc["1980-01-01":].copy() # we accept 1980 at the latest
    #without the .copy well get an error (were making two dataframes)

    # Target: Will tomorrow's close be higher than today's close?
    stock["Tomorrow"] = stock["Close"].shift(-1) 
    stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)#1 when price went up, 0 when price went down
    del stock["Tomorrow"]
    print("Accuracy using tomorrow value.")
    model.predict(stock)

    #we are comparing open price with open price. More of a predicting today approach instead of predicting tomorrow.
    stock["Target"] = (stock["Close"] > stock["Open"]).astype(int) 
    print("Accuracy using todays value.")
    model.predict(stock)


    # precision as the error metric for the algorithm (true positives / (false positives + true positives)). 
    # We do it this way as it minimizes the false negatives


    # ------------------------------------------------------------------------------------------
    # plot the stock
        # stock.plot.line(y="Close", use_index=True)
        # plt.show()