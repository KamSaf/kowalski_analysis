# Load the data and inspect it
import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(cleaned_data.head())
print(cleaned_data.info())

# Perform multiple regression analysis to predict selling price based on other variables
# Select relevant features for regression analysis
features = ["year", "condition", "odometer", "mmr"]
X = data[features]
y = data["sellingprice"]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression summary
print(model.summary())
