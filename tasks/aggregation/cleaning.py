import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/aggregation/aggregated_Capability_data.csv", delimiter=',')

# Function to clean text data
def clean_text(text):
    text = str(text).strip().lower()  # Convert to string, strip spaces, convert to lowercase
    text = text.replace('\n', ' ').replace('\r', '')  # Remove newlines
    return text

# Apply text cleaning to all text columns
text_columns = ['id', 'type', 'name', 'content', 'image', 'attachment', 'category',
       'sub_categories', 'industries', 'geo_reach', 'price_range',
       'availability_status', 'certifications', 'launch_date',
       'market_performance', 'customer_satisfaction_rate', 'innovation_level',
       'key_features', 'competitors', 'market_demand', 'requirements',
       'environmental_impact', 'sales_volume', 'support_level', 'scalability',
       'compliance_status', 'brand_reputation', 'promotion_channels',
       'additional_comments', 'likes_count', 'rating_count', 'average_rating',
       'saves_count', 'views_count']

for col in text_columns:
    if col in df.columns:  # Ensure the column exists before applying the function
        df[col] = df[col].apply(clean_text)

# Handle numeric columns, ensuring there are no non-numeric values
numeric_columns = ['likes_count', 'views_count', 'average_rating', 'saves_count', 'image', 'attachment']

for col in numeric_columns:
    if col in df.columns:  # Ensure the column exists before processing
        # Coerce errors forces invalid parsing to NaN, then fill NaN with 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Ensure no special characters in text fields
df[text_columns] = df[text_columns].replace({r'[^\x00-\x7F]+':''}, regex=True)

# Fill NaN values for specific columns with empty strings or other appropriate values
fill_values = {
    'id': '', 'type': '', 'name': '', 'content': '', 'image': '', 'attachment': '', 'category': '',
    'sub_categories': '', 'industries': '', 'geo_reach': '', 'price_range': '', 'availability_status': '',
    'certifications': '', 'launch_date': '', 'market_performance': '', 'customer_satisfaction_rate': '',
    'innovation_level': '', 'key_features': '', 'competitors': '', 'market_demand': '', 'requirements': '',
    'environmental_impact': '', 'sales_volume': '', 'support_level': '', 'scalability': '',
    'compliance_status': '', 'brand_reputation': '', 'promotion_channels': '',
    'additional_comments': '', 'likes_count': 0, 'rating_count': 0, 'average_rating': 0,
    'saves_count': 0, 'views_count': 0
}

df = df.fillna(fill_values)

# Verify the cleaned DataFrame
print("Data Types After Cleaning:\n", df.dtypes)
print("Any Missing Values:\n", df.isnull().sum())
print("Sample Data:\n", df.head())

# Export the cleaned DataFrame to a new CSV file
cleaned_file_path = "/Users/salomonmuhirwa/Documents/NACB_PRoject/aggregation/fully_102.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Fully cleaned data has been saved to {cleaned_file_path}")
