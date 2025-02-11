import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# Define the dataset folder path
folder_path = "/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/ISE/lab2/datasets/dconvert"

# Get all CSV file paths in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read all CSV files and combine them into a single DataFrame
dataframes = [pd.read_csv(file) for file in csv_files]
data = pd.concat(dataframes, ignore_index=True)  # Combine all into one DataFrame

# Check dataset structure
print("Dataset Shape:", data.shape)
print(data.head())

# Assuming the last column is the target (performance metric) and others are features
X = data.iloc[:, :-1]  # All columns except last as features
y = data.iloc[:, -1]  # Last column as target

# Store results
mape_scores = []
mae_scores = []
rmse_scores = []

# Repeat the training/testing process 30 times to avoid stochastic bias
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Store metrics
    mape_scores.append(mape)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

# Display average results
print("Average MAPE:", np.mean(mape_scores))
print("Average MAE:", np.mean(mae_scores))
print("Average RMSE:", np.mean(rmse_scores))
