import pandas as pd
import random
from faker import Faker
import datetime
import string
import json

fake = Faker()

interest = [
    "Cannabis Cultivation", "Cannabis Processing", "Cannabis Retail", 
    "Medical Cannabis", "Recreational Cannabis", "Cannabis Compliance", 
    "Cannabis Technology", "Cannabis Marketing", "Cannabis Accessories", 
    "Cannabis Education", "Cannabis Advocacy", "Cannabis Policy", 
    "Cannabis Research", "Cannabis Trends", "Cannabis Innovations", 
    "Cannabis Strains", "Cannabis Extraction Methods", "Cannabis Consumption Methods", 
    "Cannabis Edibles", "Cannabis Concentrates", "Cannabis Vaping", 
    "Cannabis Testing and Quality Control", "Cannabis Packaging", 
    "Cannabis Branding", "Cannabis Startups", "Cannabis Investment", 
    "Cannabis Industry Events", "Cannabis Conferences", "Cannabis Workshops", 
    "Cannabis Business Development", "Cannabis Regulations", "Cannabis Taxation", 
    "Cannabis Legalization", "Cannabis Supply Chain Management", 
    "Cannabis Retail Management", "Cannabis Customer Experience", 
    "Cannabis Dispensary Operations", "Cannabis Sales Strategies", 
    "Cannabis Cultivation Techniques", "Cannabis Soil and Nutrients", 
    "Cannabis Pest Management", "Cannabis Sustainability", "Cannabis Eco-Friendly Practices",
    "Cannabis Community Outreach", "Cannabis Social Responsibility", 
    "Cannabis Diversity and Inclusion", "Cannabis Health Benefits", 
    "Cannabis Pain Management", "Cannabis Mental Health", "Cannabis for Chronic Conditions", 
    "Cannabis Research and Development", "Cannabis Clinical Trials", 
    "Cannabis Drug Interactions", "Cannabis Education Programs", 
    "Cannabis Industry Certifications", "Cannabis Professional Training", 
    "Cannabis Supply Chain Innovations", "Cannabis Product Development", 
    "Cannabis Market Analysis", "Cannabis Consumer Behavior", 
    "Cannabis Industry Trends", "Cannabis Regulatory Compliance", 
    "Cannabis Intellectual Property", "Cannabis Security and Safety", 
    "Cannabis Waste Management", "Cannabis Cultivation Equipment", 
    "Cannabis Distribution Networks", "Cannabis International Markets", 
    "Cannabis Emerging Markets", "Cannabis Franchise Opportunities"
]

# Load data from JSON files
def load_json(filename):
    try:
        with open(filename, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {filename}")
        return []

industry = load_json("data/industry.json")
positions = load_json("data/position.json")
experiences = load_json("data/experience.json")
interests = load_json("data/interest.json")
education_list = load_json("data/education_list.json")
departments = load_json("data/department.json")
awards = load_json("data/award.json")
certifications = load_json("data/certification.json")
location_data = load_json("data/countries_states_cities.json")
promotion_channels = load_json("data/promotion_channel.json")
cost_cutting_priorities = load_json("data/cost_reduction_priority.json")

business_goals = load_json("data/business_goal.json")
competitors = load_json("data/competitors.json")
services = load_json("data/services.json")
products = load_json("data/product.json")
key_contacts = load_json("data/key_contact.json")
evaluation_process = load_json("data/evaluation_process.json")
performance_indicators = load_json("data/performance_indicator.json")
funding_source = load_json("data/funding_source.json")
proposal_submission_process = load_json("data/proposal_submission_process.json")

genders = ['Male', 'Female', 'Non-binary', 'Lesbian', 'Gay', 'Bisexual', 'Other']
completion_statuses = ['Completed', 'In Progress', 'Not Started']

def random_date(start, end):
    return start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

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

def generate_last_login(created_at):
    start_date = datetime.datetime(datetime.datetime.now().year, 1, 1)
    end_date = datetime.datetime.now()
    last_login = random_date(start_date, end_date)
    return last_login.strftime('%Y-%m-%d %H:%M:%S')

# Load the user accounts data
user_accounts_data = pd.read_csv('dataset/user_accounts_data.csv')

# Filter IDs that start with 'P' and get corresponding rows
p_ids_data = user_accounts_data[user_accounts_data['id'].str.startswith('U')]

# Extract relevant data
p_ids = p_ids_data['id'].tolist()
n = len(p_ids)


# Define the fixed and random components of the opportunities
pros = [f"Pro{str(i).zfill(3)}" for i in range(1, 375, 2)]  # Pro001, Pro003, Pro005, etc.
servs = [f"Serv{str(i).zfill(3)}" for i in range(2, 376, 2)]
capabilities_list = pros + servs

def generate_user_profile_data(n):
    profiles = []
    
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)
    
    for i in range(1, n+1):
        created_at = random_date(start_date, end_date)
        updated_at = random_date(created_at, end_date)
        
        num_part = f'U{i:03d}'
        letter_part = random.choice(string.ascii_uppercase)
        user_id = f'{num_part}{letter_part}'
        
        location, phone_code, latitude, longitude = select_random_location(location_data)
        
        profile = {
            "id": user_id,
            "user": random.randint(1, n),
            "bio": fake.sentence(),
            "slug": f"nacb/profile/{fake.user_name()}",
            "gender": random.choices(genders, weights=[0.5, 0.4, 0.005, 0.005, 0.005, 0.005, 0.05], k=1)[0],
            "image": fake.image_url(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y-%m-%d'),
            "address": [latitude, longitude],
            "location": location,
            "linkedin": fake.url(),
            "twitter": fake.url(),
            "personal_website": fake.url(),
            "phone_code": phone_code,
            "phone_number": fake.random_int(min=1000000000, max=9999999999),
            "industry": random.choice(industry) if industry else None,
            "position": random.choice(positions) if positions else None,
            "experiences": [random.choice(experiences) for _ in range(random.randint(1, 3))] if experiences else [],
            "interests": [random.choice(interest) for _ in range(random.randint(1, 10))] if interest else [],
            "education_level": random.choice(education_list) if education_list else None,
            "department": random.choice(departments) if departments else None,
            "availability": random.choice([True, False]),
            "awards": [random.choice(awards) for _ in range(random.randint(0, 3))] if awards else [],
            "certifications": [random.choice(certifications) for _ in range(random.randint(0, 3))] if certifications else [],
            "capabilities": ",".join(random.sample(capabilities_list, random.randint(1, 5))) if capabilities_list else "",
            
            "opportunities": ",".join(random.sample([f"OPR{str(i).zfill(3)}" for i in range(1, 121)], random.randint(1, 5))),
            "completion_status": random.choices(completion_statuses, weights=[0.6, 0.3, 0.1], k=1)[0],
            "last_login": generate_last_login(created_at),
            "created_at": created_at.strftime('%Y-%m-%d %H:%M:%S'),
            "updated_at": updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        profiles.append(profile)
    return pd.DataFrame(profiles)

# Generate user profile data
user_profiles_data = generate_user_profile_data(n)
user_profiles_data.to_csv('dataset/user_profiles_data.csv', index=False)

print("User profile data has been generated and saved to 'dataset/user_profiles_data.csv'")
