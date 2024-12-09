import pandas as pd
import random
from faker import Faker
import datetime
import string
import json

fake = Faker()

# Define probabilities for roles
roles = {
    'system_admin': 0.05,
    'nacb_admin': 0.10,
    'nacb_staff': 0.15,
    'organization_admin': 0.10,
    'organization_member': 0.30,  # High probability
    'user_professional': 0.30     # High probability
}

# Load blocked reasons from JSON file
with open('blocked_reasons.json', 'r') as file:
    blocked_reasons = json.load(file)

def generate_role():
    role = random.choices(list(roles.keys()), weights=roles.values(), k=1)[0]
    return role

def random_date(start, end):
    return start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_user_accounts_data(n):
    accounts = []
    
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime.now()
    
    for i in range(1, n+1):
        created_at = random_date(start_date, end_date)
        updated_at = random_date(created_at, end_date)
        
        # Alternate between "U00X" and "P00X" prefixes
        if i % 2 == 0:
            prefix = f"U{i//2 + 1:03d}"
        else:
            prefix = f"P{i//2 + 1:03d}"
        letter_part = random.choice(string.ascii_uppercase)
        user_id = f'{prefix}{letter_part}'
        
        blocked = random.choices([True, False], weights=[0.1, 0.9], k=1)[0]
        blocked_reason = random.choice(blocked_reasons) if blocked else None
        
        account = {
            "id": user_id,
            "username": fake.user_name(),
            "email": fake.email(),
            "password": fake.password(length=12),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "is_staff": random.choice([True, False]),
            "role": generate_role(),
            "organization": random.choice([None, random.randint(1, 100)]),  # Example organization IDs
            "blocked": blocked,
            "blocked_reason": blocked_reason,
            "last_login": random_date(start_date, end_date).strftime('%Y-%m-%d %H:%M:%S')
        }
        accounts.append(account)
    
    return pd.DataFrame(accounts)

# Generate user account data
user_accounts_data = generate_user_accounts_data(8000)  # Adjust the number as needed
user_accounts_data.to_csv('dataset/user_accounts_data.csv', index=False)

print("User account data has been generated and saved to 'dataset/user_accounts_data.csv'")
