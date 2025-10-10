# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# DataFrame
df = pd.read_csv("Virginia_Housing.csv")

# Prints
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Labels
X = df[["Square_Feet", "Bedrooms", "Bathrooms", "Year_Built", "Lot_Size", "Garage"]]
y = df["Price"]

# Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# Comparison
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Virginia House Price Prediction")
plt.grid(True)
plt.show()

# User Testing (You Enter)
sample = pd.DataFrame({
    "Square_Feet": [700],
    "Bedrooms": [8],
    "Bathrooms": [4],
    "Year_Built": [2025],
    "Lot_Size": [0.49],
    "Garage": [8]
})
predicted_price = model.predict(sample)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
