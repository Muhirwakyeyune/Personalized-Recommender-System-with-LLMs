import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, Date, TIMESTAMP, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

# Read CSV data into a DataFrame
user_data = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/organization_data.csv")

# URL encode the password
password = quote_plus('Muhirwa@@4795')

# Create a SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://salomonmuhirwa:{password}@localhost/NACB')

# Define the table schema using SQLAlchemy ORM
Base = declarative_base()
class Organization(Base):
    __tablename__ ='organizations'
    
    id =Column(String(10), primary_key=True)
    admin=Column(Integer)
    name=Column(String(100))
    email=Column(String(100))
    bio=Column(Text)
    image=Column(String(255), nullable=True)
    slug=Column(String(100))
    addresses=Column(Text)
    locations=Column(String(100))
    founded_date=Column(Date)
    website=Column(String(255), nullable=True)
    phone_code=Column(String(10), nullable=True)
    phone_number=Column(String(20), nullable=True)
    linkdin=Column(String(255), nullable=True)
    twitter=Column(String(255), nullable=True)
    industry=Column(String(100))
    cost_cutting_priorities =Column(Text, nullable=True)
    promotion_channels=Column(Text, nullable=True)
    business_goals =Column(Text, nullable=True)
    annual_revenue =Column(Float)
    competitors =Column(Text, nullable=True)
    awards=Column(Text, nullable=True)
    certifications=Column(Text, nullable=True)
    joining_reason=Column(Text, nullable=True)
    capabilities =Column(Text, nullable=True)
    opportunities=Column(Text, nullable=True)
    completion_status=Column(String(100))
    created_at=Column(TIMESTAMP)
    updated_at=Column(TIMESTAMP)
    
    
    
    

# Create the table in the database
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Import the DataFrame into the database
user_data.to_sql('organizations', engine, if_exists='replace', index=False)

# Close the session
session.close()

print("Data imported successfully.")
