import pandas as pd

# Read the CSV files
events_df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/capabilitosdata.csv")
likes_df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/like_data.csv")
ratings_df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/rating_data.csv")
saves_df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/save_data.csv")
views_df = pd.read_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/view_data.csv")

# Aggregate the data
likes_agg = likes_df[likes_df['asset_type'] == 'capability'].groupby('asset_id').size().reset_index(name='likes_count')
ratings_agg = ratings_df[ratings_df['asset_type'] == 'capability'].groupby('asset_id').agg({'id': 'count', 'rate': 'mean'}).reset_index().rename(columns={'id': 'rating_count', 'rate': 'average_rating'})
saves_agg = saves_df[saves_df['asset_type'] == 'capability'].groupby('asset_id').size().reset_index(name='saves_count')
views_agg = views_df[views_df['asset_type'] == 'capability'].groupby('asset_id').size().reset_index(name='views_count')

# Merge the aggregated data with the events DataFrame
aggregated_df = events_df.merge(likes_agg, how='left', left_on='id', right_on='asset_id', suffixes=('', '_likes'))
print("After merging likes:", aggregated_df.columns.tolist())

aggregated_df = aggregated_df.merge(ratings_agg, how='left', left_on='id', right_on='asset_id', suffixes=('', '_ratings'))
print("After merging ratings:", aggregated_df.columns.tolist())

aggregated_df = aggregated_df.merge(saves_agg, how='left', left_on='id', right_on='asset_id', suffixes=('', '_saves'))
print("After merging saves:", aggregated_df.columns.tolist())

aggregated_df = aggregated_df.merge(views_agg, how='left', left_on='id', right_on='asset_id', suffixes=('', '_views'))
print("After merging views:", aggregated_df.columns.tolist())

# Drop the duplicate columns from the merge
columns_to_drop = [col for col in aggregated_df.columns if col.startswith('asset_id')]
print("Columns to drop:", columns_to_drop)
aggregated_df = aggregated_df.drop(columns=columns_to_drop)

# Fill NaN values with 0
aggregated_df[['likes_count', 'rating_count', 'average_rating', 'saves_count', 'views_count']] = aggregated_df[['likes_count', 'rating_count', 'average_rating', 'saves_count', 'views_count']].fillna(0)

# Convert to appropriate data types
aggregated_df['likes_count'] = aggregated_df['likes_count'].astype(int)
aggregated_df['rating_count'] = aggregated_df['rating_count'].astype(int)
aggregated_df['average_rating'] = aggregated_df['average_rating'].astype(float)
aggregated_df['saves_count'] = aggregated_df['saves_count'].astype(int)
aggregated_df['views_count'] = aggregated_df['views_count'].astype(int)

# Save the final aggregated DataFrame to a CSV file for use in Tableau
aggregated_df.to_csv("/Users/salomonmuhirwa/Documents/NACB_PRoject/aggregation/aggregated_Capability_data.csv", index=False)

print("Aggregated data saved successfully.")
