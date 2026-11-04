
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.get_dummies(pd.read_csv("used_car_dataset.csv"), drop_first=True)
df.to_csv("weka_new_data.csv", index=False)

X, Y = df.drop("Price_USD", axis=1), df["Price_USD"]
x, xt, y, yt = train_test_split(X, Y, test_size=0.2, random_state=42)

m = LinearRegression().fit(x, y)
p = m.predict(xt)

print("\nIntercept (b0):")
print(m.intercept_)
print("\nCoefficients:")
for f, c in zip(X.columns, m.coef_):
    print(f, "=", round(c, 3))

e = f"Price = {round(m.intercept_, 2)}"
for f, c in zip(X.columns, m.coef_):
    e += f" + ({round(c, 2)})*{f}"

print("\nLinear Regression Equation:")
print(e)
print("\nR2 Score:", round(r2_score(yt, p), 4))
print("Mean Squared Error:", round(mean_squared_error(yt, p), 2))

r = pd.DataFrame({"Actual Price": yt, "Predicted Price": p})
print("\nSample Predictions:")
print(r.head(10))

plt.figure(figsize=(8, 6))
plt.scatter(yt, p)
plt.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted Values")
plt.grid(True)
plt.show()
