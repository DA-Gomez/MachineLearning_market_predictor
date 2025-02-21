import dataSetup as stock

watchlist = {
    "S&P500": "^GSP",
    "Tesla": "TSLA",
}
for key in watchlist.keys():
    print(key)
userInput = input("Select a stock/index: ")

for key in watchlist.keys():
    if key == userInput:
        stock.analyze(watchlist[key])
        break
else:
    print("Doesnt exist")
