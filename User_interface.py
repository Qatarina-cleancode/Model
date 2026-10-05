import streamlit as st
import joblib
import pandas as pd

#Create the user interface
st.set_page_config(page_title="Diabetes_Risk_Predictor")
st.title("Diabetes Risk Analysis Tool")
st.write("Fill in te vitals below") 

#Load the model
model=joblib.load('diabetes_model.joblib')
st.sidebar.header("Patient Vitals")

#sliders/icons
glucose = st.sidebar.slider('Glucose Level (mg)', min_value=20,max_value=90, value=50)
BMI=st.sidebar.slider('BMI(bmi)', min_value=20,max_value=100, value=70)
age=st.sidebar.slider('age(years)',min_value=1,max_value=100,value=50)
steps=st.sidebar.number_input('steps(km/m)',min_value=100,max_value=5000)


#Load all the variables into the dataframe
input_data=pd.DataFrame({
    'glucose_level':[glucose],
    'BMI':[BMI],
    'age':[age],
    'Daily_steps':[steps]

})
#Display the patients summary
st.subheader('patients vital summary')
col1,col2,col3,col4= st.columns(4)
col1.metric("glucose", f"{glucose} mg")
col2.metric("BMI", f"{BMI} bmi")
col3.metric("age", f"{age} years")
col4.metric("Daily_steps",f"{steps}km/m")

#Prediction button
if st.button("Run Risk Prediction", type="secondary"):
    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0][1] * 100

    st.markdown("................")
    st.subheader("Dialysis Results")
    #Display te results based on prediction
    if prediction==1:
        st.error(f"HIGH RISK OF DIABETES (Risk score:{probability:})")
        st.warning("Recoomendation:Immediate visit to the hospital")

    else:
        st.success(f"LOW RISK/NORMAL(RiskScore:{probability:.1f}%)")
        st.info("Recommendation: Maintain regular annual wellness checkups")