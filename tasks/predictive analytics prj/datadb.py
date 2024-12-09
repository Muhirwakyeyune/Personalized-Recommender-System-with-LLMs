import psycopg2

import pandas as pd

user_data=pd.read_csv("/Users/salomonmuhirwa/Desktop/NACB/fake data/users_data.csv")






# connection

pgcon=psycopg2.connect(host='localhost',
                       user='salomonmuhirwa',
                       password='Muhirwa@@4795',
                       database='NACB')
# #cursor
pgcursor=pgcon.cursor()

# required code
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 
pgcon.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) 


# drop db
pgcursor.execute('DROP DATABASE IF EXISTS NACB')
# create db
pgcursor.execute('CREATE DATABASE NACB')


pgcon.commit()

#close
pgcon.close()



