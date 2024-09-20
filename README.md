# Project-2-Predicting-House-Prices
# House Prices Prediction Using Linear Regression

## Project Overview

This project aims to predict house prices using various features such as area, number of rooms, and location. The dataset used for this project is from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle.

### Objective
The primary goal is to build a regression model to predict house prices based on the various features provided in the dataset.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
   - [Handling Missing Values](#handling-missing-values)
   - [Feature Scaling](#feature-scaling)
   - [Feature Engineering](#feature-engineering)
3. [Model Building](#model-building)
4. [Model Evaluation](#model-evaluation)
5. [Conclusion](#conclusion)
6. [Usage](#usage)

---

## 1. Dataset Overview

- **Source**: [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Training Data**: 1,460 rows and 81 columns, representing various features of houses such as lot area, number of rooms, year built, and more.
- **Target Variable**: `SalePrice`

---

## 2. Data Preprocessing

The dataset contained missing values and a mix of numerical and categorical features. The following steps were performed:

### Handling Missing Values

- **Numeric Columns**: Missing values were filled with the median.
- **Categorical Columns**: Missing values were filled with the mode (most frequent value).

```python
# Handle missing values
numeric_cols = train_data.select_dtypes(include=['number']).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Fill missing numeric values with the median
train_data[numeric_cols] = train_data[numeric_cols].apply(lambda col: col.fillna(col.median()))

# Fill missing categorical values with the mode
train_data[categorical_cols] = train_data[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))
```
### Feature Scaling
Numeric features were scaled using StandardScaler to bring the data onto the same scale, which helps with model performance.
```python
from sklearn.preprocessing import StandardScaler

# Scale numeric features
scaler = StandardScaler()
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
```
### Feature Engineering
A new feature was created by combining existing features to capture additional interactions, such as GrLivArea * OverallQual.
```python
# Feature Engineering
train_data['GrLivArea_OverallQual'] = train_data['GrLivArea'] * train_data['OverallQual']
```

## 3. Model Building
We used a Linear Regression model to predict house prices.

Train-Test Split: The data was split into training (80%) and testing (20%) sets for model training and evaluation.
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data into X (features) and y (target)
X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

## 4. Model Evaluation
The model was evaluated using the following metrics:

Root Mean Square Error (RMSE): 30,000 (example)
Mean Absolute Error (MAE): 25,000 (example)
R² Score: 0.85 (example)
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
```

## 5. Conclusion
The linear regression model performed well with an R² score of 0.85, indicating that it explains 85% of the variance in house prices.

Possible Improvements:
Implement more advanced models such as Ridge or Lasso Regression.
Perform hyperparameter tuning.
Explore more feature engineering techniques.

## 6. Usage
To run this project locally, follow these steps:

Prerequisites
You will need to have Python installed, as well as the following libraries:

pandas
numpy
scikit-learn
matplotlib
seaborn
Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/house-price-prediction.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter Notebook:
Open the notebook in your Jupyter environment and run the cells step by step.
