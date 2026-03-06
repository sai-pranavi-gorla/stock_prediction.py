# Stock Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Download stock data from Yahoo Finance
stock = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Step 2: Keep only Close price
data = stock[['Close']]

# Step 3: Create prediction column
data['Prediction'] = data[['Close']].shift(-30)

# Step 4: Create feature dataset
X = np.array(data.drop(['Prediction'], axis=1))
X = X[:-30]

# Step 5: Create target dataset
y = np.array(data['Prediction'])
y = y[:-30]

# Step 6: Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 7: Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 8: Model accuracy
accuracy = model.score(x_test, y_test)
print("Model Accuracy:", accuracy)

# Step 9: Predict future prices
future_days = 30
x_future = data.drop(['Prediction'], axis=1).tail(future_days)
x_future = np.array(x_future)

prediction = model.predict(x_future)

print("\nFuture Predictions:")
print(prediction)

# Step 10: Plot graph
plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title("Stock Price History")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
