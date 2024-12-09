import pandas as pd

df=pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/event_data.csv")



# Convert dates and times to appropriate formats
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S', errors='coerce').dt.time
df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S', errors='coerce').dt.time

# Handle missing values
df.fillna({'author': 'Unknown', 'author_name': 'Unknown', 'end_date': '9999-12-31', 'end_time': '23:59:59'}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save the cleaned data to a new CSV file
df.to_csv('dataset/cleaned_event_data.csv', index=False)