import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
from sklearn.model_selection import train_test_split

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


loaded_model=joblib.load('random_forest_churn_model.pkl')
#predict on test set

#features and target
features=['annual_revenue','founded_days','tenure']
X=df[features]
y=df['completion_status']


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)


y_pred=loaded_model.predict(X_test)

# accuracy
accuracy=accuracy_score(y_test,y_pred)

print(f'Accuracy: {accuracy:.2}')