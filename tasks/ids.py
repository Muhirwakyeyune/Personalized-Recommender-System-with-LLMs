import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/tweets.csv", delimiter=',')

# Function to clean text data
def clean_text(text):
    text = str(text).strip().lower()  # Convert to string, strip spaces, convert to lowercase
    text = text.replace('\n', ' ').replace('\r', '')  # Remove newlines
    return text

# Apply text cleaning to all text columns
text_columns = ['id', 'slug', 'user', 'content', 'published_date', 'location',
       'hashtags', 'mentions', 'urls', 'images_urls', 'videos_urls',
       'num_of_likes', 'num_of_retweets', 'num_of_replies', 'num_of_quotes',
       'score', 'type', 'parent_tweet_id', 'topics', 'categories',
       'created_at']


print(df.columns)

for col in text_columns:
    df[col] = df[col].apply(clean_text)

# Handle numeric columns, ensuring there are no non-numeric values
numeric_columns = ['num_of_likes', 'num_of_retweets', 'num_of_replies', 'num_of_quotes', 'score', 'images_urls', 'videos_urls']

for col in numeric_columns:
    # Coerce errors forces invalid parsing to NaN, then fill NaN with 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Ensure no special characters in text fields
df[text_columns] = df[text_columns].replace({r'[^\x00-\x7F]+':''}, regex=True)

# Fill any remaining NaN values with default values
# Fill any remaining NaN values with default values
df = df.fillna({
    'id': '',
    'slug': '', 
    'user': '', 
    'content': '', 
    'published_date': '', 
    'location': '', 
    'hashtags': '', 
    'mentions': '', 
    'urls': '', 
    'images_urls': '', 
    'videos_urls': '', 
    'num_of_likes': 0, 
    'num_of_retweets': 0, 
    'num_of_replies': 0, 
    'num_of_quotes': 0, 
    'score': 0, 
    'type': '', 
    'parent_tweet_id': '', 
    'topics': '', 
    'categories': '', 
    'created_at': ''
})


# Verify the cleaned DataFrame
print("Data Types After Cleaning:\n", df.dtypes)
print("Any Missing Values:\n", df.isnull().sum())
print("Sample Data:\n", df.head())

# Export the cleaned DataFrame to a new CSV file
cleaned_file_path = "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/fully_103.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Fully cleaned data has been saved to {cleaned_file_path}")
