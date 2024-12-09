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

with open("data/evaluation_process.json", "r") as file:
    evaluation_process = json.load(file)  # This is a duplicate, it can be removed

with open("data/perfomance_indicators.json", "r") as file:
    performance_indicators = json.load(file)  # This is a duplicate, it can be removed

# Print loaded data for verification (optional)
print(len(industry))
print(len(positions))
print(len(experiences))
print(len(interests))
print(len(education_list))
print(len(departments))
print(len(awards))
print(len(certifications))
print(len(location_data))
print(len(promotion_channels))
print(len(cost_cutting_priorities))
print(len(capabilities))
print(len(business_goals))
print(competitors)
print(services)
print(products)
print(key_contacts)
print(evaluation_process)
print(performance_indicators)
print(funding_source)
print(proposal_submission_process)
