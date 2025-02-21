from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score #did it actually go up when the target is 1 
import pandas as pd

def predict(stock):

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) 
    #n_estimators being the number of individual number of decision trees (higher val == more accuracy)
    #min_samples_split protects about overfitting
    #form my understading, random_state makes it so we get the same result if we run it again and again


    #make the model train on the past stock prices
    train = stock.loc["1980-01-01":"2022-01-01"].dropna()   
    test = stock.loc["2022-01-02":].dropna()  
    #all of the rows up to except the last 100 in the training set, and the last 100 rows in the test set

    predictors = ["Close", "Volume", "Open", "High", "Low"] 
    #be careful when using all the rows as it might seem like a good dataset, it wont really predict. Dont use tomorrow data because in reality the model wont know tomorrows price
    model.fit(train[predictors], train["Target"]) #trains_model(predictor_columns, predict target)


    #accuracy measurment
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index) #predictions are in a numpy array, so we convert to pandas series

    precision_score(test["Target"], preds)
    # combined = pd.concat([test["Target"], preds], axis = 1) #concatinating the test target and the predicted values (axis=1 means put inputs as a column)
    # combined.plot()

    #more predictors
    horizons = [5, 20, 60] #horizons on which we want to look at rolling averages of 2 days, 5 days, 60 days ...
    # [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = stock.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        stock[ratio_column] = stock["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column] # panda will return NaN when there isnt enough data for the #"horizon" average

    stock = stock.dropna()

    #backtesting
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1) 

    def Predict(train, test, predictors, model): #we want to change the return so its a probability and not a true 1 and false 0
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors]) [:,1] #this is just matlab type stuff, we are getting the second column (the probability the stock price goes up)
        #threshold of 60%
        preds[preds >= .6] = 1
        preds[preds < .6] = 0

        preds = pd.Series(preds, index = test.index, name = "Predictions")
        combined = pd.concat([test["Target"], preds], axis = 1)
        return combined

    def Backtest(data, model, predictors, start=2500, step=250): #start=2500 is taking 10 years of data and predicting the 11th year (step=250), then 11 years of data to predict 12 and so on
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i + step)].copy()

            predictions = Predict(train, test, predictors, model)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)

    predictions = Backtest(stock, model, new_predictors) # new_predictors is used because ratios are better than solid numbers

    # print(f"{predictions["Predictions"].value_counts()} \n")

    print(precision_score(predictions["Target"], predictions["Predictions"]))

    stock.plot.line(y="Close", use_index=True)
    # predictions["Target"].value_counts() / predictions.shape[0]

    return stock
