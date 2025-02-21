# MachineLearning_market_predictor

As of Jan 2025, the ai predicts the s&p 500 using downloaded data and backtesting (how good the model is). I chose the s&p 500 due to its low volatility and predictibility (from experience).


I will use the yahoo finance api to import market data

I will predict if the price will go up or down tomorrow not the actual price

``pip install yfinance``

``pip install matplotlib``

``pip install scikit-learn`` (model used)

https://www.geeksforgeeks.org/how-to-import-yfinance-as-yf-in-python/ 

https://ranaroussi.github.io/yfinance/index.html 

## Model Info

RandomForestClassifier trains individual decision trees with randomized parameters. And then averaging those result (resistant to overfit and picks up non linear tendencies in the data)

This model is not currently reliable as it only uses time series data. It could give you long term gains as it gives out accurate results 60% of the time.

This model is also for the long term as of now.

## ToDo

- Add in news or current economic state to the model so it analyses if it is a safe time to buy (interest rates, inflation, etc)

- Add in other indeces like nasdaq or qqq to make more informed predictions

- Add more scope, use monthly, weekly, hourly data

Maybe Ill be able to use this bot to trade futures and options.

Some other thing I would like to do is track nancy pelosi portfolio and see the success rate 

Feb 2025
    Ive implemented stocks. We dont take into account the post market which is fine unless we are talkinga bout earnings. The next step foward is to implement earnings.

    Another next step is to disregard economic crashes

### References

https://github.com/dataquestio/project-walkthroughs/tree/master/sp_500

https://github.com/dataquestio/project-walkthroughs/blob/master/stock/StockProject.ipynb 