import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import joblib
# URl encode the password

password=quote_plus('Muhirwa@@4795')
# Create a SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://salomonmuhirwa:{password}@localhost/NACB')


#load from the database
query='SELECT * FROM organizations'
df=pd.read_sql_query(query, engine)

#features and target
features=['annual_revenue','founded_days','tenure']
X=df[features]
y=df['completion_status']


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#model training

model=RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


#predict churn

# Predict churn
df['churn_probability'] = model.predict_proba(X)[:, 1]
df['predicted_churn_status'] = df['churn_probability'].apply(lambda x: 'Churn' if x > 0.5 else 'No Churn')

# Save predictions back to SQL
df[['id', 'churn_probability', 'predicted_churn_status']].to_sql('organization_churn_predictions', engine, if_exists='replace', index=False)


joblib.dump(model,"random_forest_churn_model.pkl")

print("Model saved")