import pandas as pd
import random
from faker import Faker
import datetime

fake = Faker()

# Helper function to generate random dates
def random_date(start, end):
    return start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Load the user profiles data
user_profiles_data = pd.read_csv('dataset/user_profiles_data.csv')
organization_data = pd.read_csv('dataset/organization_data.csv')

# Combine user IDs from both datasets
user_ids = user_profiles_data['id'].tolist()
organization_ids = organization_data['id'].tolist()

# Define the request statuses
request_statuses = ['pending', 'accepted', 'rejected']

def generate_connect_data(n):
    connects = []
    
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    
    for i in range(n):
        created_at = random_date(start_date, end_date)
        updated_at = random_date(created_at, end_date)
        
        sender = random.choice(user_ids)
        receiver = sender
        # Ensure receiver is different from sender
        while receiver == sender:
            receiver = random.choice(user_ids + organization_ids)
        
        connect = {
            "id": i + 1,
            "sender": sender,
            "receiver": receiver,
            "status": random.choices(request_statuses, weights=[0.4, 0.5, 0.1]),
            "created_at": created_at.strftime('%Y-%m-%d %H:%M:%S'),
            "updated_at": updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        connects.append(connect)
    return pd.DataFrame(connects)

# Generate connect data
num_records = 1000  # Adjust the number of records as needed
connect_data = generate_connect_data(num_records)
connect_data.to_csv('dataset/connect_data.csv', index=False)

print("Connect data has been generated and saved to 'dataset/connect_data.csv'")
