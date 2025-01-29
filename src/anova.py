# Load the data and inspect it
import pandas as pd
import scipy.stats as stats

# Load the dataset
file_path = "car_prices.csv"
data = pd.read_csv(file_path)
cleaned_data = data.dropna(subset=["condition", "odometer", "mmr"])
# Display the first few rows and summary of the dataset
print(cleaned_data.head())
print(cleaned_data.info())


# Perform ANOVA analysis to check if there are significant differences in selling price based on car make

# Group data by 'make' and extract selling prices
groups = [group["sellingprice"].dropna() for name, group in data.groupby("body")]

# Perform one-way ANOVA
anova_result = stats.f_oneway(*groups)

# Display the ANOVA result
print("Wynik analizy ANOVA:")
print("F:", anova_result.statistic)
print("p:", anova_result.pvalue)
