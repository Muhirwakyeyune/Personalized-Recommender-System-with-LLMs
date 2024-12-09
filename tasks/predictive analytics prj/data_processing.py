import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime

# URL encode the password
password = quote_plus('Muhirwa@@4795')

# Create a SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://salomonmuhirwa:{password}@localhost/NACB')

# Load data from the database
query = "SELECT * FROM organizations"
df = pd.read_sql(query, engine)


# Clean the `completion_status` column
df['completion_status'] = df['completion_status'].str.strip("[]").str.strip("'")


# Verify cleaned values
print("Cleaned unique values:", df['completion_status'].unique())

# Map the values to numeric
status_mapping = {
    'Complete': 1,
    'Incomplete': 0
}

df['completion_status']=df['completion_status'].map(status_mapping)
df['completion_status'].fillna(0, inplace=True)

# Feature Engineering
df['founded_date'] = pd.to_datetime(df['founded_date'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['updated_at'] = pd.to_datetime(df['updated_at'])

df['founded_days'] = (datetime.now() - df['founded_date']).dt.days
df['tenure'] = (df['updated_at'] - df['created_at']).dt.days



# Handling missing values
df['annual_revenue'].fillna(0, inplace=True)

# Save the updated DataFrame back to the database
df.to_sql('organizations', engine, if_exists='replace', index=False)

print("Feature engineering completed and data updated successfully.")
