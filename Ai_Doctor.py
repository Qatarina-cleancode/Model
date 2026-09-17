#import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Generate seed data
np.random.seed(42)
n=400 #Generate 400 records
#stimulate the features
glucose=np.random.normal(loc=110,scale=25,size=n).round(2)
BMI=np.random.normal(loc=25,scale=5,size=n)
age=np.random.randint(20,80,size=n)
steps=np.random.randint(1500,12000,size=n)
#Make the Biological risk_score
Risk_score=(glucose*0.04)+(BMI*0.08)+(age*0.03)-(steps*0.0003)
y_target=(Risk_score>6.5).astype(int)

#combine into a dataframe
df=pd.DataFrame({'glucose_level':glucose,
                 'age':age,
                 'BMI':BMI,
                 'Steps_walked':steps,
                 'Diabetes_Risk':y_target})
print('---Score_Risk---')
print(df.head())
