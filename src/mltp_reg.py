# Multiple Regression Analysis: Predicting Number of Injuries
# Selecting relevant features for regression
features = [
    "Visibility Level",
    "Number of Vehicles Involved",
    "Speed Limit",
    "Driver Alcohol Level",
    "Driver Fatigue",
    "Pedestrians Involved",
    "Cyclists Involved",
    "Emergency Response Time",
    "Traffic Volume",
]
X = df[features]
y = df["Number of Injuries"]

# Handling missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Adding a constant for statsmodels
X = sm.add_constant(X)

# Fitting the regression model
model = sm.OLS(y, X).fit()

# Summary of the regression model
print(model.summary())
