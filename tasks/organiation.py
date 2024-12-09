import pandas as pd
import random
from faker import Faker
import datetime
import json

fake = Faker()

# Load the user profiles data to get the 'user' column
user_profiles_data = pd.read_csv('dataset/user_profiles_data.csv')
user_ids = user_profiles_data['user']
n = len(user_ids)  # Set n to the length of user_ids

# Load data from JSON files
with open("data/industry.json", "r") as json_file:
    industry = json.load(json_file)

with open("data/position.json", "r") as json_file:
    positions = json.load(json_file)

with open("data/experience.json", "r") as json_file:
    experiences = json.load(json_file)

with open("data/interest.json", "r") as json_file:
    interests = json.load(json_file)

with open("data/education_list.json", "r") as json_file:
    education_list = json.load(json_file)

with open("/Users/salomonmuhirwa/Documents/NACB_PRoject/data/department.json", "r") as json_file:
    departments = json.load(json_file)

with open("data/award.json", "r") as json_file:
    awards = json.load(json_file)

with open("data/certification.json", "r") as json_file:
    certifications = json.load(json_file)

with open("data/countries_states_cities.json", "r") as file:
    location_data = json.load(file)

with open("data/promotion_channel.json", "r") as file:
    promotion_channels = json.load(file)

with open("data/cost_reduction_priority.json", "r") as file:
    cost_cutting_priorities = json.load(file)

with open("data/capabilities.json", "r") as file:
    capabilities = json.load(file)

with open("data/business_goal.json", "r") as file:
    business_goals = json.load(file)

with open("data/competitors.json", "r") as file:
    competitors = json.load(file)

with open("data/services.json", "r") as file:
    services = json.load(file)

with open("data/product.json", "r") as file:
    products = json.load(file)

with open("data/key_contact.json", "r") as file:
    key_contacts = json.load(file)

with open("data/evaluation_process.json", "r") as file:
    evaluation_process = json.load(file)

with open("data/performance_indicator.json", "r") as file:
    performance_indicators = json.load(file)

with open("data/funding_source.json", "r") as file:
    funding_source = json.load(file)

with open("data/proposal_submission_process.json", "r") as file:
    proposal_submission_process = json.load(file)

def select_random_location(location_data):
    country = random.choice(list(location_data.keys()))
    country_info = location_data[country]
    
    # Ensure 'states' is a dictionary and not empty
    if "states" in country_info and country_info["states"]:
        state_name, state_info = random.choice(list(country_info["states"].items()))
        location = f"{country}, {state_name}"
        latitude = state_info['latitude']
        longitude = state_info['longitude']
    else:
        location = country  # Use only the country name if no states are available
        latitude = country_info['latitude']
        longitude = country_info['longitude']
    
    phone_code = country_info["phone_code"]
    return location, phone_code, latitude, longitude

def generate_organization_data(n):
    locations, phone_codes, latitudes, longitudes = zip(*[select_random_location(location_data) for _ in range(n)])
    
    # Define the fixed and random components of the opportunities
    pros = [f"Pro{str(i).zfill(3)}" for i in range(1, 375, 2)]  # Pro001, Pro003, Pro005, etc.
    servs = [f"Serv{str(i).zfill(3)}" for i in range(2, 376, 2)]
    capabilities = pros + servs
    
    opportunities = [f"OPR{str(i).zfill(3)}" for i in range(1, 121)]  # OPR001, OPR002, OPR003, etc.

    data = {
        "id": [f"ORG{str(i).zfill(3)}" for i in range(1, n+1)],
        "admin": [random.choice(user_ids) for _ in range(n)],
        "name": [fake.company() for _ in range(n)],
        "email": [fake.company_email() for _ in range(n)],
        "bio": [fake.catch_phrase() for _ in range(n)],
        "image": [None for _ in range(n)],  # Assuming image will be uploaded later
        "slug": [fake.slug() for _ in range(n)],
        "addresses": list(zip(latitudes, longitudes)),
        "locations": locations,
        "founded_date": [fake.date() for _ in range(n)],
        "website": [fake.url() for _ in range(n)],
        "phone_code": phone_codes,
        "phone_number": [fake.msisdn() for _ in range(n)],
        "linkedin": [fake.url() for _ in range(n)],
        "twitter": [fake.url() for _ in range(n)],
        "industry": [random.choice(industry) for _ in range(n)] if industry else [None] * n,
        "cost_cutting_priorities": [json.dumps(random.sample(cost_cutting_priorities, random.randint(1, 3))) for _ in range(n)],
        "promotion_channels": [json.dumps(random.sample(promotion_channels, random.randint(1, 4))) for _ in range(n)],
        "business_goals": [json.dumps(random.sample(business_goals, random.randint(1, 4))) for _ in range(n)],
        "annual_revenue": [round(random.uniform(1e5, 1e9), 2) for _ in range(n)],
        "competitors": [json.dumps(random.sample(competitors, random.randint(1, 4))) for _ in range(n)],
        "awards": [json.dumps(random.sample(awards, random.randint(0, 3))) for _ in range(n)],
        "certifications": [json.dumps(random.sample(certifications, random.randint(0, 3))) for _ in range(n)],
        "joining_reason": [fake.catch_phrase() for _ in range(n)],
        "capabilities": [",".join(random.sample(capabilities, random.randint(1, 5))) for _ in range(n)],
        "opportunities": [",".join(random.sample(opportunities, random.randint(1, 5))) for _ in range(n)],
        "completion_status": [random.choices(['Complete', 'Incomplete'], weights=[0.7, 0.3])[0] for _ in range(n)],
    }

    # Generate created_at and updated_at with constraints
    created_ats = []
    updated_ats = []
    for _ in range(n):
        created_at = fake.date_time_this_decade()
        # Ensure updated_at is not before created_at
        updated_at = fake.date_time_between_dates(datetime_start=created_at)
        created_ats.append(created_at)
        updated_ats.append(updated_at)
        
    data["created_at"] = created_ats
    data["updated_at"] = updated_ats

    return pd.DataFrame(data)

# Generate organization data
organization_data = generate_organization_data(n)  # Generating data for n organizations
organization_data.to_csv('dataset/organization_data.csv', index=False)
