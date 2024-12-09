import pandas as pd
import random
from faker import Faker
import json

fake = Faker()

import json

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


# Load the user accounts data
user_accounts_data = pd.read_csv('dataset/user_accounts_data.csv')

# Filter IDs that start with 'P' and get corresponding rows
p_ids_data = user_accounts_data[user_accounts_data['id'].str.startswith('P')]

# Extract relevant data
p_ids = p_ids_data['id'].tolist()
emails = p_ids_data['email'].tolist()
first_names = p_ids_data['first_name'].tolist()
last_names = p_ids_data['last_name'].tolist()

# Number of IDs
n = len(p_ids)

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

# Function to generate industry contacts data
def industry_contacts(ids, emails, first_names, last_names):
    locations = [select_random_location(location_data) for _ in range(n)]
    
    phone_numbers = [fake.phone_number() for _ in range(n)]
    organizations = [fake.company_suffix() + ' ' + str(random.randint(100, 999)) for _ in range(n)]  # Example organization names
    roles = [random.choice(['Manager', 'Director', 'Assistant', 'Consultant']) for _ in range(n)]
    
    slugs = [f"{first_names[i].lower()}-{last_names[i].lower()}" for i in range(n)]
    industry_types = [random.choice(['ventures','service_providers','investors','academics','policy_makers','non_profit']) for _ in range(n)]  # Example industries
    availabilities = [{'days': random.sample(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], k=random.randint(1, 7))} for _ in range(n)]
    
    # Define the fixed and random components of the opportunities
    pros = [f"Pro{str(i).zfill(3)}" for i in range(1, 375, 2)]  # Pro001, Pro003, Pro005, etc.
    servs = [f"Serv{str(i).zfill(3)}" for i in range(2, 376, 2)]
    capabilities = pros + servs
    
    data = {
        "id": ids,  # Use the filtered IDs
        "email": emails,  # Copy the corresponding emails
        "phone_code": [location[1] for location in locations],  # Extract phone codes for each location
        "phone_number": phone_numbers,
        "first_name": first_names,
        "last_name": last_names,
        "slug": slugs,
        "organization": organizations,
        "type": industry_types,
        "industry": [random.choice(industry) for _ in range(n)] if industry else [],  # Ensure length matches n
        "location": [location[0] for location in locations],  # Extract location names
        "address": [[location[2], location[3]] for location in locations],  # Extract latitudes and longitudes
        "num_of_employees": [random.randint(1, 500) for _ in range(n)],
        "role": roles,
        "linkedin": [fake.url() for _ in range(n)],
        "twitter": [fake.url() for _ in range(n)],
        "personal_website": [fake.url() for _ in range(n)],
        "experiences": [random.choice(experiences) for _ in range(n)] if experiences else [],
        "awards": [random.choice(awards) for _ in range(n)] if awards else [],
        "certifications": [random.choice(certifications) for _ in range(n)] if certifications else [],
        "education_level": [random.choice(education_list) for _ in range(n)] if education_list else [],
        
        "promotion_channels": [random.choice(promotion_channels) for _ in range(n)] if promotion_channels else [],
        "cost_cutting_priorities": [random.choice(cost_cutting_priorities) for _ in range(n)] if cost_cutting_priorities else [],
        "interests": [random.choice(interests) for _ in range(n)] if interests else [],
        "availability": availabilities,
        "capabilities": [",".join(random.sample(capabilities, random.randint(1, 5))) for _ in range(n)],
        "opportunities": [",".join(random.sample([f"OPR{str(i).zfill(3)}" for i in range(1, 121)], random.randint(1, 5))) for _ in range(n)],

        "business_goals":[random.choice(business_goals) for _ in range(n)] if business_goals else [],
        "notes": [json.dumps([fake.sentence() for _ in range(random.randint(1, 2))]) for _ in range(n)],
    }
    
    return pd.DataFrame(data)

# # Generate industry contacts data
# industry_contacts_data = industry_contacts(p_ids, emails, first_names, last_names)
# industry_contacts_data.to_csv('dataset/industry_contacts.csv', index=False)



data=pd.read_csv("dataset/industry_contacts.csv")

print(data.head)