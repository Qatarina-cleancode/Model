#import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
                 'Daily_steps':steps,
                 'Diabetes_Risk':y_target})
print('---Score_Risk---')
print(df.head())
#separate features and the target
x=df[['glucose_level','BMI','age','Daily_steps']]
y=df['Diabetes_Risk']

#Train/Split 80% used to train the Ai model, 20% for testing 
x_train, x_test,y_train,y_test=train_test_split(
x,y,
test_size=0.20,random_state =42)

#Train and build the random forest
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train, y_train)
#Evaluate the model
y_pred=model.predict(x_test)
print("---Model Diagonistic")
print( f"Overall Accuracy is: " f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
print("confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\n classification Model:\n",classification_report(y_test,y_pred))


 

