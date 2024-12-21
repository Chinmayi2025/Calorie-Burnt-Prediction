import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# Load the datasets
df1 = pd.read_csv("calories.csv")
df2 = pd.read_csv("exercise.csv")

# Merge datasets
df = pd.concat([df2, df1["Calories"]], axis=1)
df.drop(columns=["User_ID"], inplace=True)

# Convert categorical data
df["Gender"] = pd.get_dummies(df["Gender"], drop_first=True)

# Define features and target
X = df.drop(columns=["Calories"], axis=1)
y = df["Calories"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train and save the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save the trained model
with open("calorie_model.pickle", "wb") as file:
    pickle.dump(model, file)

print("Model training complete and saved as calorie_model.pickle.")
