import pandas as pd

# Load the opportunity data
opportunity_data = pd.read_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/opportunity_dataa.csv')

# Load interaction data
like_data = pd.read_csv('dataset/like_data.csv')
view_data = pd.read_csv('dataset/view_data.csv')
rating_data = pd.read_csv('dataset/rating_data.csv')
comment_data = pd.read_csv('dataset/comment_data.csv')
save_data = pd.read_csv('dataset/save_data.csv')

# Filter interaction data by 'opportunity' asset type
like_opportunity = like_data[like_data['asset_type'] == 'opportunity']
view_opportunity = view_data[view_data['asset_type'] == 'opportunity']
rating_opportunity = rating_data[rating_data['asset_type'] == 'opportunity']
comment_opportunity = comment_data[comment_data['asset_type'] == 'opportunity']
save_opportunity = save_data[save_data['asset_type'] == 'opportunity']

# Aggregate data
like_agg = like_opportunity.groupby('asset_id').size().reset_index(name='like_count')
view_agg = view_opportunity.groupby('asset_id').size().reset_index(name='view_count')
rating_agg = rating_opportunity.groupby('asset_id')['rate'].mean().reset_index(name='average_rating')
comment_agg = comment_opportunity.groupby('asset_id').size().reset_index(name='comment_count')
save_agg = save_opportunity.groupby('asset_id').size().reset_index(name='save_count')

# Merge aggregated data with opportunity_data
aggregated_data = opportunity_data.merge(like_agg, left_on='id', right_on='asset_id', how='left')
aggregated_data = aggregated_data.merge(view_agg, on='asset_id', how='left')
aggregated_data = aggregated_data.merge(rating_agg, on='asset_id', how='left')
aggregated_data = aggregated_data.merge(comment_agg, on='asset_id', how='left')
aggregated_data = aggregated_data.merge(save_agg, on='asset_id', how='left')

# Fill NaN values with 0 (for counts) or with a sensible default (e.g., 0 or neutral rating for averages)
aggregated_data.fillna({'like_count': 0, 'view_count': 0, 'average_rating': 0, 'comment_count': 0, 'save_count': 0}, inplace=True)

# Drop the 'asset_id' column as it's now redundant
aggregated_data.drop(columns=['asset_id'], inplace=True)

# Save the aggregated data to a new CSV file
aggregated_data.to_csv('/Users/salomonmuhirwa/Documents/NACB_PRoject/dataset/aggregated_opportunity_data.csv', index=False)

print(aggregated_data.head())
