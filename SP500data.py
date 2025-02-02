import yfinance as yf #yahoo finance api
import matplotlib.pyplot as plt 

sp500 = yf.Ticker("^GSPC") #ticker class (downloads price history for the GSPC symbol)
sp500 = sp500.history(period="max") 
# queries historical prices (each row is the price for each trading day. Date, Open, High, Low, close, Volume, dividents, stocks splits)


sp500.plot.line(y="Close", use_index=True)
# plt.show()

del sp500["Dividends"]
del sp500["Stock Splits"]

# predict if the price will go up or down tomorrow not the actual price
sp500["Tomorrow"] = sp500["Close"].shift(-1) #shifts the current price to yesterday's "tomorrows" price

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)#1 when price went up, 0 when price went down

sp500 = sp500.loc["1990-01-01":].copy() #without the .copy well get an error (were making two dataframes)