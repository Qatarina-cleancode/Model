import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# creating User Interface, we rendered in CSS
st.markdown("""
<style>
div.stButton > Button:first-child {
    background-color: #F20c00;
    Color:white;
    font-size:18px;
    font-weight:bold;
    boarder-radius:8px;
    padding: 18px, 20px;
    widith: 100%;

}

[data-testid="stMetric"] {
        background-color:#70F3FF;
        padding: 12px;
        border-radius: 8px;
        # border-left: 4px solid #F20c00;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩸 Diabetes Risk Assessment Tool")
st.write("Enter patient vitals below to get real-time clinical Risk Prediction.")

# Load the pre-trained Model
model = joblib.load('diabetes_model.joblib')
st.sidebar.header("Patient Intake Vitals")

# Interactive Sliders and inputs for the user
glucose = st.sidebar.slider("glucose_level (mg/dL)", min_value=70, max_value=200, value=120)
bmi = st.sidebar.slider("BMI)", min_value=15.0, max_value=50.0, value=25.0)
age = st.sidebar.slider("age (Years)", min_value=5, max_value=100, value=18)
steps = st.sidebar.number_input("Daily_steps", min_value=500, max_value=20000, value=5000)

# Organize inputs into a dataframe matching training features
input_data = pd.DataFrame({
    'glucose_level': [glucose],
    'BMI': [bmi],
    'age': [age],
    'Daily_steps': [steps]
})
# Display patient summary
st.subheader("Patient's vitals summary")
col1,col2,col3,col4=st.columns(4)
col1.metric("glucose_level",f"{glucose} gm")
col2.metric("BMI", f"{bmi}bmi")
col3.metric("age",f"{age}yrs")
col4.metric("Daily_steps", f"{steps}km/m")

#prediction button
if st.button("Run Risk Analysis",type="primary"):
    prediction =model.predict(input_data)[0]
    probability =model.predict_proba(input_data)[0][1]*100

#split out results and visuals
    res_col1, res_col2=st.columns([1,1])
    with res_col1:

       st.subheader("✔ Diagnosis Results")
       st.write(f"**Overall Calculated Risk:** `{probability:.1f}%`")
       st.progress(int(probability))

#results based on prediction
    if prediction == 1:
        st.error(f"🚨 HIGH RISK OF DIABETES (Risk Score: {probability:.1f}%)")
        st.warning("Recommendation: Schedule immediate fasting plasma glucose (FPG) test and HbA1c Screening.")
    else:
        st.success(f"✅ LOW RISK / NORMAL (Risk Score: {probability:.1f}%)")
        st.info("Recommendation: Maintain regular annual wellness checkups and healthy physical activity.")
    with res_col2:
        st.subheader("Model Feature Importance")
        
        feature_names = ['glucose_Level', 'BMI', 'age', 'Daily_steps']
        importances=model.feature_importances_*100
        df=pd.DataFrame({"Feature":feature_names, "Importance (%)":importances })
        df=df.sort_values("Importance (%)",ascending=False)

        fig, ax = plt.subplots(figsize=(5, 2.8))
            
        bar_color = "#eb0f0f" if prediction == 1 else "#0FF822"
        bars = ax.barh(df['Feature'], df['Importance (%)'], color=bar_color)
            
            # Annotate bars with percentage values
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=8)

            ax.set_xlim(0, max(df['Importance (%)']) + 15)
            ax.set_xlabel('Influence (%)', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        