import pandas as pd
import random
from faker import Faker
from datetime import date, timedelta
import json

fake = Faker()

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

with open("data/requirement.json", "r") as file:
    requirement = json.load(file)

with open("data/key_contact.json", "r") as file:
     key_contact = json.load(file)

# with open("data/performance_indicators.json", "r") as file:
#     performance_indicators = json.load(file)


type_opportunity = ['consulting', 'rfp', 'job', 'partnership', 'sponsorship', 'volunteering', 'investment', 'purchase']

opportunity_durations = [
    "<1 month", "1-3 months", "3-6 months", "6-12 months", ">1 year"
]

budgets = [
    "$0", "<1000", "$1000-$5000", "$5000-$10000", "$10000-$50000", ">$50000"
]

expected_outcomes = ["high", "medium", "low"]
strategic_importances = ["high", "medium", "low"]
market_impacts = ["high", "medium", "low"]
risk_levels = ["high", "medium", "low"]
statuses = ["open", "pending", "closed"]
engagement_levels = ["high", "medium", "low"]
contract_types = ["fixed", "variable", "mixed"]
selection_criteria_list = [
    "Relevant industry experience",
    "Proven track record of success",
    "Strong communication skills",
    "Innovative approach",
    "Financial stability",
    "Ability to meet deadlines",
    "Technical expertise",
    "Quality of proposal",
    "Cost-effectiveness",
    "Customer references",
    "Alignment with strategic goals",
    "Sustainability initiatives",
    "Team qualifications",
    "Past performance",
    "Risk management strategy"
]

categories = {
    "Consulting": ["IT Consulting", "Management Consulting", "Financial Consulting", "HR Consulting"],
    "Procurement": ["Technology Procurement", "Office Supplies", "Professional Services"],
    "Employment": ["Full-time", "Part-time", "Contract"],
    "Partnership": ["Strategic Partnership", "Joint Venture", "Co-branding"],
    "Sponsorship": ["Event Sponsorship", "Program Sponsorship", "Scholarship Sponsorship"],
    "Volunteering": ["Community Service", "Skill-based Volunteering", "Pro-bono Consulting"],
    "Investment": ["Seed Funding", "Series A Funding", "Venture Capital"],
    "Purchase": ["Office Furniture", "Computer Equipment", "Software Licenses"]
}

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
    
    phone_code = country_info["phone_code"]
    return location, phone_code, latitude, longitude

def generate_opportunity(n):
    opportunities = []

    for i in range(n):
        location, phone_code, latitude, longitude = select_random_location(location_data)
        start_date = fake.date_between_dates(date_start=date(2024, 3, 1), date_end=date(2024, 6, 30))
        end_date = start_date + timedelta(days=random.randint(1, 360))
        type_ = random.choice(type_opportunity)
        category = random.choice(list(categories.keys()))
        sub_categories = random.choices(categories[category], k=random.randint(1, len(categories[category])))

        opportunity = {
            "id": f"OPR{str(i + 1).zfill(3)}",
            "type": type_,
            "name": f"{random.choice(['Consulting Engagement', 'RFP for Services', 'Job Opening', 'Strategic Partnership', 'Sponsorship', 'Volunteering Opportunity', 'Investment', 'Purchase'])} - {category}",
            "slug": fake.slug(),
            "content": fake.text(),
            "image": None,
            "attachment": None,
            "category": category,
            "sub_categories": random.choice(sub_categories),
            "industry": random.choice(industry),
            "application_deadline": fake.date_between_dates(date_start=start_date, date_end=end_date),
            "duration": random.choice(opportunity_durations),
            "requirements": ",".join(random.sample(requirement, random.randint(2, 4))),
            "budgets": random.choice(budgets),
            "funding_source": ",".join(random.sample(funding_source, random.randint(1, 3))),
            "location": location,
            "key_contacts": ",".join(random.sample(key_contact, random.randint(1, 3))),
            "Expected_outcomes": random.choice(expected_outcomes),
            "strategic_importance": random.choice(strategic_importances),
            "market_impact": random.choice(market_impacts),
            "risk_level": random.choice(risk_levels),
            "status": random.choice(statuses),
            "engagement_levels": random.choice(engagement_levels),
            "partners_involved": fake.company(),
            "proposal_submission_process": ",".join(random.sample(proposal_submission_process, random.randint(1, 5))),
            "selection_criteria": ",".join(random.sample(selection_criteria_list, random.randint(3, 5))),
            "evaluation_process": ",".join(random.sample(evaluation_process, random.randint(3, 5))),
            "contract_type": random.choice(contract_types),
            "performance_indicators": ",".join(random.sample(performance_indicators, random.randint(1, 2))),
            "additional_comment": fake.sentence()
        }
        opportunities.append(opportunity)
    
    return pd.DataFrame(opportunities)

opportunity_data = generate_opportunity(120)
opportunity_data.to_csv("dataset/opportunity_dataa.csv", index=False)
