
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *

df = pd.read_csv("house_price_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

df["Furnishing_status"] = LabelEncoder().fit_transform(df["Furnishing_status"])
X, y = df.drop("Price_Lakhs", axis=1), df["Price_Lakhs"]

a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(a))
print("Testing samples :", len(b))

m = DecisionTreeRegressor(criterion="squared_error", max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42).fit(a, c)
p = m.predict(b)

print("\n--- Model Evaluation (80-20 split) ---")
print("MSE :", round(mean_squared_error(d, p), 3))
print("RMSE :", round(np.sqrt(mean_squared_error(d, p)), 3))
print("MAE :", round(mean_absolute_error(d, p), 3))
print("R2 :", round(r2_score(d, p), 3))

t = DecisionTreeRegressor(criterion="squared_error", max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42).fit(a, c)

plt.figure(figsize=(22, 12))
plot_tree(t, feature_names=X.columns, filled=True, rounded=True, precision=1, fontsize=11)
plt.title("CART Regression Tree Structure (depth=3, for visualization)")
plt.tight_layout()
plt.savefig("regression_tree.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved plot: regression_tree.png")

r = []
for name, size in [("50-50", 0.5), ("60-40", 0.4), ("80-20", 0.2)]:
    x1, x2, y1, y2 = train_test_split(X, y, test_size=size, random_state=42)
    m = DecisionTreeRegressor(criterion="squared_error", max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42).fit(x1, y1)
    p = m.predict(x2)
    mse = mean_squared_error(y2, p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y2, p)
    print("\n" + "="*60)
    print("Case:", name, "Split")
    print("Training samples :", len(x1))
    print("Testing samples :", len(x2))
    print(f"\nMSE : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R2 Score : {r2_score(y2,p):.4f}")
    print(f"R2 Percentage : {r2_score(y2,p)*100:.2f} %")
    print("\nSample Predictions (Actual vs Predicted):")
    print(pd.DataFrame({"Actual": y2.values[:5], "Predicted": np.round(p[:5], 2)}).to_string(index=False))
    r.append([name, len(x1), len(x2), round(mse,3), round(rmse,3), round(mae,3)])

print("\nSummary: Performance Across All Train-Test Splits")
print(pd.DataFrame(r, columns=["Split", "Train_size", "Test_size", "MSE", "RMSE", "MAE"]).to_string(index=False))
