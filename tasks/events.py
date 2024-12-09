import pandas as pd
import random
from faker import Faker
import json
import datetime

fake = Faker()

# Define possible categories and entry types
news_categories = ["Politics", "Business & Finance", "Technology", "Health", "Science", "Entertainment", "Sports"]


entry_types = ["auto", "manual"]
week_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

with open("data/countries_states_cities.json", "r") as file:
    location_data = json.load(file)

def select_random_location(location_data):
    country = random.choice(list(location_data.keys()))
    country_info = location_data[country]
    
    latitude = country_info['latitude']
    longitude = country_info['longitude']
    phone_code = country_info['phone_code']
    
    return country, phone_code, latitude, longitude

def generate_event_data(n):
    locations, phone_codes, latitudes, longitudes = zip(*[select_random_location(location_data) for _ in range(n)])

    data = {
        "id": [f"EV{str(i+1).zfill(3)}" for i in range(n)],
        "title": [fake.catch_phrase() for _ in range(n)],
        "source": [random.choice(["NACB", fake.domain_name()]) for _ in range(n)],
        "url": [fake.url() for _ in range(n)],
        "summary": [fake.sentence() for _ in range(n)],
        "slug": [fake.slug() for _ in range(n)],
        "entry_type": [random.choice(entry_types) for _ in range(n)],
        "author": [random.randint(1, 100) if random.choice(entry_types) == "manual" else None for _ in range(n)],
        "content": [fake.text() for _ in range(n)],
        "author_name": [fake.name() for _ in range(n)],
        "author_image": [None for _ in range(n)],  # Assuming image URLs will be added later
        "content_vector": [None for _ in range(n)],  # Placeholder for actual vectors
        "categories": [random.choice(news_categories) for _ in range(n)],
        "tags": [random.sample(news_categories, random.randint(1, 5)) for _ in range(n)],
        "address": [fake.address() for _ in range(n)],
        "location": locations,
        "week_day": [random.choice(week_days) for _ in range(n)],
        "start_date": [fake.date_this_year() for _ in range(n)],
        "start_time": [fake.time() for _ in range(n)],
        "end_date": [fake.date_this_year() if random.choice([True, False]) else None for _ in range(n)],
        "end_time": [fake.time() if random.choice([True, False]) else None for _ in range(n)],
        "created_at": [fake.date_time_this_decade() for _ in range(n)],
        "updated_at": [fake.date_time_this_decade() for _ in range(n)],
    }
    return pd.DataFrame(data)

# Generate event data
event_data = generate_event_data(500)  # Generating data for 100 events
event_data.to_csv('dataset/event1_data.csv', index=False)
