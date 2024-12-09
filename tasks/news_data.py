import pandas as pd
import random
import datetime
from faker import Faker
import json
# Initialize Faker
fake = Faker()



entry_type=["Auto","Manual"]
# Define lists for titles, categories, and tags
news_titles = [
    "Breaking News: Major Cannabis Legislation Passed",
    "Cannabis Industry Boom: What You Need to Know",
    "New Trends in Cannabis Cultivation for 2024",
    "How Cannabis Technology is Shaping the Future",
    "Exploring the Benefits of Cannabis for Health",
    "Top Cannabis Products to Watch This Year",
    "Cannabis Market Analysis: Key Insights and Trends",
    "Regulatory Changes in the Cannabis Industry",
    "Innovations in Cannabis Extraction Methods",
    "Cannabis Retail Strategies for Increased Sales"
]

categories = [
    "Legislation", "Market Analysis", "Cultivation", "Technology", "Health",
    "Products", "Trends", "Regulations", "Extraction", "Retail"
]

tags = [
    "Cannabis", "Legislation", "Technology", "Health", "Market Trends",
    "Products", "Innovation", "Regulations", "Cultivation", "Retail"
]


def select_random_location(location_data):
    country = random.choice(list(location_data.keys()))
    country_info = location_data[country]
    
    if "states" in country_info and country_info["states"]:
        state_name, state_info = random.choice(list(country_info["states"].items()))
        location = f"{country}, {state_name}"
        latitude = state_info['latitude']
        longitude = state_info['longitude']
    else:
        location = country
        latitude = country_info['latitude']
        longitude = country_info['longitude']
    
    phone_code = country_info.get("phone_code", "")
    return location, phone_code, latitude, longitude

def generate_news_data(n, location_data):
    news_list = []
    
    for i in range(1, n + 1):
        news_id = f"N{i:03d}"
        title = random.choice(news_titles)
        slug = title.lower().replace(" ", "-").replace(":", "").replace(",", "")
        source = "NACB"
        url = f"https://example.com/{slug}"
        summary = fake.sentence()
        entry_type_choice = random.choice(entry_type)  # Assuming entry_type as 'News'
        author_id = None
        content = fake.paragraph()
        author_name = fake.name()
        author_image = fake.image_url()
        content_vector = [random.random() for _ in range(100)]  # Example vector
        categories_assigned = random.choice(categories)
        tags_assigned = random.sample(tags, k=random.randint(1, 5))
        address, phone_code, latitude, longitude = select_random_location(location_data)
        week_day = fake.day_of_week()
        published_at = fake.date_this_year()
        created_at = fake.date_time_this_decade()
        updated_at = fake.date_time_this_decade()

        news = {
            "id": news_id,
            "title": title,
            "source": source,
            "url": url,
            "summary": summary,
            "slug": slug,
            "entry_type": entry_type_choice,
            "author": author_id,
            "content": content,
            "author_name": author_name,
            "author_image": author_image,
            "content_vector": content_vector,
            "categories": categories_assigned,
            "tags": tags_assigned,
            "address": f"{latitude}, {longitude}",
            "location": address,
            "week_day": week_day,
            "published_at": published_at,
            "created_at": created_at,
            "updated_at": updated_at
        }
        
        news_list.append(news)
    
    return news_list

# Load location data
def load_json(filename):
    try:
        with open(filename, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {filename}")
        return {}

location_data = load_json("data/countries_states_cities.json")

# Generate and save news data
news_data = generate_news_data(500, location_data)
news_df = pd.DataFrame(news_data)

# Save to CSV
news_df.to_csv("dataset/news_data.csv", index=False)

print("News data has been generated and saved to 'dataset/news_data.csv'")
