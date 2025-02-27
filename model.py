from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score #did it actually go up when the target is 1 
import pandas as pd


def Predict(train, test, predictors, model): # returns a probability
    model.fit(train[predictors], train["Target"])

    #make predictions
    preds = model.predict_proba(test[predictors]) [:,1] #this is just matlab type stuff, we are getting the second column (the probability the stock price goes up)
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    #Threshold
    preds[preds >= .6] = 1
    preds[preds < .6] = 0

    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

def Backtest(data, model, predictors, start=2500, step=250): #start=2500 is taking 10 years of data and predicting the 11th year (step=250), then 11 years of data to predict 12 and so on
    train = data.loc["1980-01-01":"2022-01-01"].dropna() #training set
    test = data.loc["2022-01-02":].dropna()  #test set
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()

        predictions = Predict(train, test, predictors, model)

        all_predictions.append(predictions)
    return pd.concat(all_predictions)


#predictors
def averages(stock, check):
    intervals = [5, 22, 60, 180] #averages of 5 days, 22 days, 60 days ...
    # [2, 5, 60, 250, 1000]
    new_predictors = []

    for interval in intervals:
        rolling_averages = stock.rolling(interval).mean()

        ratio_column = f"Close_Ratio_{interval}"
        stock[ratio_column] = stock["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{interval}"
        stock[trend_column] = stock.shift(1).rolling(interval).sum()["Target"]
        # stock = stock.dropna() causes SettingWithCopyWarning 

        new_predictors += [ratio_column, trend_column] # panda will return NaN when there isnt enough data for the #"horizon" average
    return new_predictors

def trendAverages(stock, predictors): # model can get a taste of the current market trends
    weekly_mean = stock.rolling(7).mean()
    quarterly_mean = stock.rolling(90).mean()
    annual_mean = stock.rolling(365).mean()
    weekly_trend = stock.shift(1).rolling(7).mean()["Target"]

    stock["weekly_mean"] = weekly_mean["Close"] / stock["Close"]
    stock["quarterly_mean"] = quarterly_mean["Close"] / stock["Close"]
    stock["annual_mean"] = annual_mean["Close"] / stock["Close"]

    stock["annual_weekly_mean"] = stock["annual_mean"] / stock["weekly_mean"]
    stock["annual_quarterly_mean"] = stock["annual_mean"] / stock["quarterly_mean"]
    stock["weekly_trend"] = weekly_trend

    stock["open_close_ratio"] = stock["Open"] / stock["Close"]
    stock["high_close_ratio"] = stock["High"] / stock["Close"]
    stock["low_close_ratio"] = stock["Low"] / stock["Close"]

    new_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
    return new_predictors

def predict(stock, check):
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) 
    #n_estimators being the number of individual number of decision trees (higher val == more accuracy)
    #min_samples_split protects about overfitting
    #form my understading, random_state makes it so we get the same result if we run it again and again


    #make the model train on the past stock prices
    # train = stock.loc["1980-01-01":"2022-01-01"].dropna() #training set
    # test = stock.loc["2022-01-02":].dropna()  #test set
    #51% perdiction rate
    # model.fit(train[predictors], train["Target"]) #trains_model(predictor_columns, predict target) #be careful when using all the rows as it might seem like a good dataset, it wont really predict.
    # preds = model.predict(test[predictors])
    # preds = pd.Series(preds, index = test.index) #predictions are in a numpy array, so we convert to pandas series
    # precision_score(test["Target"], preds)
    # combined = pd.concat([test["Target"], preds], axis = 1) #concatinating the test target and the predicted values (axis=1 means put inputs as a column)
    # combined.plot()

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    
    predictions1 = Backtest(stock, model, averages(stock)) 
    print(precision_score(predictions1["Target"], predictions1["Predictions"]), end="")
    if check: 
        return 

    # get actual closing price and shift prices a day
    data = stock[["Close"]]
    data = data.rename(columns={'Close': 'ActualClose'}) 
    # identify if price went up or down and then shift foward one day, so we are predicting tomorrow from today data
    data["Target"] = stock.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    stock = stock.shift(1)
    stock = data.join(stock[predictors]).iloc[1:]

    predictions2 = Backtest(stock, model, trendAverages(stock, predictors)) 
    print(precision_score(predictions2["Target"], predictions2["Predictions"]))

    # visualizes the number of predictions
    # print(predictions1["Predictions"].value_counts())
    # print(predictions2["Predictions"].value_counts())
