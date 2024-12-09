
Copy code
import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()


# Function to generate organizations data
def generate_organizations_data(n):
    data = {
        "Organization_ID": [random.randint(1, 10**8) for i in range(1, n+1)],
        "Name": [fake.company() for _ in range(n)],
        "Industry": [fake.bs() for _ in range(n)],
        "Size": [random.choice(["Small", "Medium", "Large"]) for _ in range(n)],
        "Country": [fake.country() for _ in range(n)],
        "State": [fake.state() for _ in range(n)],
        "Founded_Year": [fake.year() for _ in range(n)],
        "Revenue": [random.randint(100000, 10000000) for _ in range(n)],
        "Number_of_Employees": [random.randint(10, 500) for _ in range(n)],
        "Capabilities": [",".join(random.sample(capabilities_list,random.randint(1,3))) for _ in range(n)],
        "Key_Contacts": [fake.name() for _ in range(n)],
        "Membership_Level": [random.choice(["Basic", "Premium"]) for _ in range(n)],
        "Needs": [",".join(random.sample(needs_list,random.randint(3,6))) for _ in range(n)],
        "Strategic_Partners": [fake.company() for _ in range(n)],
        "Certification": [",".join(random.sample(awards_certificates_list,random.randint(0,4))) for _ in range(n)],
        
        
        "Trainings": [",".join(random.sample(trainings_list,random.randint(0,5))) for _ in range(n)],
        "Awards_and_Certificates": [",".join(random.sample(awards_certificates_list,random.randint(0,4))) for _ in range(n)],
        "Key_Projects": [",".join(random.sample(cannabis_projects,(random.randint(0,4)))) for _ in range(n)],
        "Business_Model": [random.choice(["B2B", "B2C", "B2G"]) for _ in range(n)],
        "Market_Presence": [random.choice(["Local", "Regional", "Global"]) for _ in range(n)],
        "Sustainability_Initiatives": [fake.text(max_nb_chars=100) for _ in range(n)],
        "Regulatory_Compliance": [random.choice(["Compliant", "Non-compliant"]) for _ in range(n)],
        "Technological_Adoption": [random.choice(["Low", "Medium", "High"]) for _ in range(n)],
        "Customer_Satisfaction_Rate": [random.choice(["Yes", "No"]) for _ in range(n)],
        "Growth_Rate": [random.uniform(0.5, 10.0) for _ in range(n)],
        "Type_of_Social_Media": [random.choice(["LinkedIn", "Twitter", "Facebook", "Telegram"]) for _ in range(n)],
        "Employee_Retention_Rate": [random.uniform(70.0, 100.0) for _ in range(n)],
        "Community_Engagement": [random.choice(["One-way communication", "Two-way communication", "Collaborative", "Collaborative action"]) for _ in range(n)],
        
        
        
    }
    return pd.DataFrame(data)

# Generate organizations data
organizations_data = generate_organizations_data(5000)  # Generating data for 5000 organizations
organizations_data.to_csv('organizations_data.csv', index=False)