import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the data
df = pd.read_csv("road_accident_dataset.csv")

# Select numerical columns for analysis
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
df_numerical = df[numerical_cols]

print("Numerical columns selected for analysis:")
print(numerical_cols)


# One-way ANOVA test comparing Number of Injuries across different Road Types
road_type_groups = [
    group for _, group in df.groupby("Pedestrians Involved")["Number of Injuries"]
]
f_stat, p_val = stats.f_oneway(*road_type_groups)

print("One-way ANOVA Results for Pedestrians Involved vs Number of Injuries:")
print("F-statistic:", f_stat)
print("p-value:", p_val)

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.boxplot(x="Pedestrians Involved", y="Number of Injuries", data=df)
plt.xticks(rotation=45)
plt.title("Number of Injuries by Pedestrians Involved")
plt.tight_layout()
plt.show()
