import pandas as pd
import random
from faker import Faker

fake = Faker()

# Load the user profiles data
user_profiles_data = pd.read_csv('dataset/user_accounts_data.csv')
user_ids = user_profiles_data['id']

# Load the organization data
organization_data = pd.read_csv('dataset/organization_data.csv')
organization_ids = organization_data['id']

# Load the industry contacts data
industry_contacts_data = pd.read_csv('dataset/user_accounts_data.csv')
industry_contacts_ids = industry_contacts_data['id']

# Combine all IDs into a single list
all_user_ids = list(user_ids) + list(organization_ids) + list(industry_contacts_ids)

# Load the datasets for asset IDs
news_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/news_data.csv')
event_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/event1_data.csv')
capability_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/capabilito_data.csv')
opportunity_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/opportunity_dataa.csv')
tweet_user_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/tweet_user_data.csv')

# Extract asset IDs
news_ids = news_data['id'].tolist()
event_ids = event_data['id'].tolist()
capability_ids = capability_data['id'].tolist()
opportunity_ids = opportunity_data['id'].tolist()
tweet_ids = tweet_user_data['tweet_id'].tolist()

# Asset Types
asset_types = ['news', 'event', 'tweet', 'opportunity', 'capability']

# Function to get random asset_id based on asset_type
def get_random_asset_id(asset_type):
    if asset_type == 'news':
        return random.choice(news_ids)
    elif asset_type == 'event':
        return random.choice(event_ids)
    elif asset_type == 'capability':
        return random.choice(capability_ids)
    elif asset_type == 'opportunity':
        return random.choice(opportunity_ids)
    elif asset_type == 'tweet':
        return random.choice(tweet_ids)

# Generate data functions
def generate_data(n, asset_types, data_type):
    data = {
        "id": [fake.unique.random_number(digits=18) for _ in range(n)],  # Unique bigints
        "user": [random.choice(all_user_ids) for _ in range(n)],
        "asset_type": [random.choice(asset_types) for _ in range(n)],
        "asset_id": [],
        "created_at": [fake.date_time_this_decade() for _ in range(n)]
    }
    for asset_type in data["asset_type"]:
        data["asset_id"].append(get_random_asset_id(asset_type))
    if data_type == 'rating':
        data["rate"] = [random.randint(1, 5) for _ in range(n)]
    return pd.DataFrame(data)

# Number of records to generate for each table
num_records = 20000

# Generate data for each table
like_data = generate_data(num_records, asset_types, 'like')
view_data = generate_data(num_records, asset_types, 'view')
rating_data = generate_data(num_records, asset_types, 'rating')
comment_data = generate_data(num_records, asset_types, 'comment')
save_data = generate_data(num_records, asset_types, 'save')

# Save data to CSV files
like_data.to_csv('dataset/like_data.csv', index=False)
view_data.to_csv('dataset/view_data.csv', index=False)
rating_data.to_csv('dataset/rating_data.csv', index=False)
comment_data.to_csv('dataset/comment_data.csv', index=False)
save_data.to_csv('dataset/save_data.csv', index=False)
