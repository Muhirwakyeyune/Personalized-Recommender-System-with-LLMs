import pandas as pd

# Load the dataset
capability_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/capabilito_data.csv')

# Select relevant columns
cleaned_data = capability_data[['id', 'type', 'name', 'category', 'industries', 'geo_reach', 'market_demand', 'sales_volume', 'brand_reputation']]

# Drop rows with missing values in critical columns
cleaned_data.dropna(subset=['id', 'type', 'name', 'sales_volume', 'brand_reputation'], inplace=True)

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('cleaned_capability_data.csv', index=False)
