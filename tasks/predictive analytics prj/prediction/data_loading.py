import pandas as pd

# Define the file paths
file_paths = [
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/industry_contacts.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/user_accounts_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/event_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/opportunity_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/organization_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/rating_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/organization_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/capability_data.csv",
    "/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/tweet_user_data.csv"
]

# Read and display the first five rows of each CSV file
for path in file_paths:
    data = pd.read_csv(path)
    print(f"Displaying the first five rows of {path}:")
    print(data.head())
    print("\n")
