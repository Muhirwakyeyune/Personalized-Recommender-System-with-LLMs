import pandas as pd
import random
from faker import Faker
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

fake = Faker()

key_feature = [
    "High Quality", "Cost-Effective", "Scalable", "Sustainable", "Innovative", 
    "Customizable", "User-Friendly", "Fast Delivery", "Reliable", "Secure", 
    "Environmentally Friendly", "Energy Efficient", "Advanced Technology", 
    "Comprehensive Support", "Certified", "Flexible", "Industry-Compliant", 
    "High Performance", "Durable", "Low Maintenance", "Mobile-Friendly", 
    "Cloud-Based", "Data-Driven", "AI-Integrated", "Multi-Language Support", 
    "High Capacity", "Compact Design", "High Precision", "Low Risk", 
    "Easy Integration"
]

requirement = [
    "High-Speed Internet", "Cloud Storage", "Database Management System", 
    "Mobile Device Compatibility", "Desktop Application", "Web Application", 
    "API Integration", "IoT Integration", "Blockchain Technology", 
    "Machine Learning Algorithms", "Artificial Intelligence", "Virtual Reality", 
    "Augmented Reality", "Cybersecurity Measures", "Data Encryption", 
    "Biometric Authentication", "Multi-Factor Authentication", "Backup and Recovery Systems", 
    "Real-Time Data Processing", "Big Data Analytics", "Cross-Platform Compatibility", 
    "High-Performance Computing", "Edge Computing", "Quantum Computing", 
    "Software Development Kit (SDK)", "Hardware Compatibility", "DevOps Tools", 
    "Continuous Integration/Continuous Deployment (CI/CD)", "Containerization (Docker, Kubernetes)", 
    "Serverless Architecture", "Minimum Experience Level", "Entry-Level", 
    "Mid-Level", "Senior-Level", "Executive-Level", "Educational Qualifications", 
    "High School Diploma", "Associate's Degree", "Bachelor's Degree", 
    "Master's Degree", "Doctorate", "Certifications", "Industry-Specific Certifications", 
    "Professional Certifications", "Technical Certifications", "Technical Skills", 
    "Programming Languages", "Software Proficiency", "Data Analysis", 
    "Project Management Tools", "Cybersecurity Knowledge", "Soft Skills", 
    "Communication Skills", "Leadership Skills", "Teamwork", "Problem-Solving", 
    "Time Management", "Work Authorization", "Work Visa", "Citizenship", 
    "Permanent Residency", "Travel Requirements", "Willingness to Travel", 
    "No Travel Required", "Occasional Travel", "Frequent Travel", 
    "Language Proficiency", "English", "Spanish", "French", "German", 
    "Mandarin", "Other (Specify)", "Physical Requirements", "Ability to Lift a Certain Weight", 
    "Standing for Long Periods", "Working in Various Weather Conditions", 
    "Manual Dexterity", "Security Clearances", "Background Check", 
    "Security Clearance Level 1", "Security Clearance Level 2", 
    "Security Clearance Level 3", "Specific Industry Experience", 
    "Minimum Years in Industry", "Experience with Specific Equipment", 
    "Experience in Similar Projects", "Software/Hardware Requirements", 
    "Specific Software Knowledge", "Specific Hardware Proficiency", 
    "Experience with Cloud Services", "Availability", "Immediate Availability", 
    "Available within 1 Month", "Available within 3 Months", "Flexible Availability", 
    "Budget Management Experience", "Experience Managing Budgets up to $10,000", 
    "Experience Managing Budgets up to $50,000", "Experience Managing Budgets up to $100,000", 
    "Experience Managing Budgets Over $100,000", "Project Management Experience", 
    "Experience Leading Teams", "Experience Managing Multiple Projects", 
    "Experience with Agile Methodologies", "Experience with Waterfall Methodologies", 
    "Location Preferences", "Local Candidates Only", "National Candidates", 
    "International Candidates", "Specific Equipment Operation", 
    "Machinery Operation", "Laboratory Equipment", "Heavy Equipment", 
    "Specialized Tools", "Client Management Experience", 
    "Experience with Client Presentations", "Experience with Client Negotiations", 
    "Experience with Client Retention"
]

categories = [
    "Cultivation", "Processing", "Retail", "Medical", "Recreational", 
    "Technology", "Compliance", "Marketing", "Accessories", "Education"
]

sub_categories = {
    "Cultivation": ["Indoor Growing", "Outdoor Growing", "Greenhouse Growing", "Hydroponics", "Organic Farming", "Genetics and Breeding"],
    "Processing": ["Extraction", "Infusion", "Packaging", "Labeling", "Quality Control", "Edibles"],
    "Retail": ["Dispensaries", "Online Sales", "Delivery Services", "Point of Sale Systems"],
    "Medical": ["Pharmaceuticals", "Therapeutic Applications", "CBD Products", "Patient Care", "Clinical Trials"],
    "Recreational": ["THC Products", "Adult-Use Products", "Event Organizing", "Social Consumption Spaces"],
    "Technology": ["Software Solutions", "Hardware Solutions", "IoT and Automation", "Data Analytics", "Security Systems"],
    "Compliance": ["Regulatory Compliance", "Licensing", "Auditing", "Legal Services"],
    "Marketing": ["Branding", "Advertising", "Public Relations", "Social Media Management", "Market Research"],
    "Accessories": ["Consumption Devices", "Storage Solutions", "Apparel", "Merchandise"],
    "Education": ["Training Programs", "Workshops and Seminars", "Online Courses", "Research Publications", "Industry Reports"]
}

price_range = ["$0", "<1000", "$1000-$5000", "$5000-$10000", "$10000-$50000", ">$50000"]
availability_status = ["Available", "Unavailable", "Limited"]
market_performance = ["High", "Medium", "Low"]
customer_satisfaction_rate = ["High", "Medium", "Low"]
innovation_level = ["High", "Medium", "Low"]
market_demand = ["High", "Medium", "Low"]
environmental_impact = ["High", "Medium", "Low"]
sales_volume = ["High", "Medium", "Low"]
support_level = ["High", "Medium", "Low"]
scalability = ["High", "Medium", "Low"]
compliance_status = ["Compliant", "Non-Compliant"]
brand_reputation = ["High", "Medium", "Low"]

def get_random_countries(location_data, min_countries=1, max_countries=6):
    num_countries = random.randint(min_countries, max_countries)
    country_names = list(location_data.keys())
    selected_countries = random.sample(country_names, min(num_countries, len(country_names)))
    return selected_countries

def generate_capability_data(n):
    data = {
        "id": [],
        "type": [],
        "name": [],
        "content": [fake.text() for _ in range(n)],
        "image": [None for _ in range(n)],
        "attachment": [None for _ in range(n)],
        "category": [],
        "sub_categories": [],
        "industries": [json.dumps(random.sample(industry, random.randint(1, 3))) for _ in range(n)],
        "geo_reach": [json.dumps(get_random_countries(location_data, 1, 6)) for _ in range(n)],
        "price_range": [random.choice(price_range) for _ in range(n)],
        "availability_status": [random.choice(availability_status) for _ in range(n)],
        "certifications": [json.dumps(random.sample(certifications, random.randint(0, 3))) for _ in range(n)],
        "launch_date": [fake.date() for _ in range(n)],
        "market_performance": [random.choice(market_performance) for _ in range(n)],
        "customer_satisfaction_rate": [random.choice(customer_satisfaction_rate) for _ in range(n)],
        "innovation_level": [random.choice(innovation_level) for _ in range(n)],
        "key_features": [json.dumps(random.sample(key_feature, random.randint(1, 3))) for _ in range(n)],
        "competitors": [json.dumps(random.sample(competitors, random.randint(1, 4))) for _ in range(n)],
        "market_demand": [random.choice(market_demand) for _ in range(n)],
        "requirements": [json.dumps(random.sample(requirement, random.randint(1, 3))) for _ in range(n)],
        "environmental_impact": [random.choice(environmental_impact) for _ in range(n)],
        "sales_volume": [random.choice(sales_volume) for _ in range(n)],
        "support_level": [random.choice(support_level) for _ in range(n)],
        "scalability": [random.choice(scalability) for _ in range(n)],
        "compliance_status": [random.choice(compliance_status) for _ in range(n)],
        "brand_reputation": [random.choice(brand_reputation) for _ in range(n)],
        "promotion_channels": [json.dumps(random.sample(promotion_channels, random.randint(1, 4))) for _ in range(n)],
        "additional_comments": [json.dumps([fake.text() for _ in range(random.randint(0, 3))]) for _ in range(n)]
    }

    for i in range(n):
        if random.choice([True, False]):
            # Generate odd Pro IDs
            capability_id = f"Pro{str(2 * i + 1).zfill(3)}"
            capability_type = "Product"
            capability_name = random.choice(products)
        else:
            # Generate even Serv IDs
            capability_id = f"Serv{str(2 * i + 2).zfill(3)}"
            capability_type = "Service"
            capability_name = random.choice(services)

        category = random.choice(categories)
        sub_category = json.dumps(random.sample(sub_categories[category], random.randint(1, 3)))

        data["id"].append(capability_id)
        data["type"].append(capability_type)
        data["name"].append(capability_name)
        data["category"].append(category)
        data["sub_categories"].append(sub_category)

    return pd.DataFrame(data)

# Load the user profiles data to get the 'id' column
user_profiles_data = pd.read_csv('dataset/user_profiles_data_new.csv')
ids = user_profiles_data['id'].tolist()

# Generate capability data
capability_data = generate_capability_data(376)  # Generating data for capabilities with IDs
capability_data.to_csv('dataset/capability_data.csv', index=False)
