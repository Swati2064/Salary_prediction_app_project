import streamlit as st
import pickle 
import numpy as np

#load saved model
model = pickle.load(open(r"D:\VS_CODE\ML_Practical\liner_regression_model.pkl", "rb"))

st.title("Salary Prediction App")

st.write("This app predicts salary based on experience")

years_experience=st.number_input("Enter your years of experience",min_value=0.0,max_value=50.0,value=1.0, step=0.5)

#when button is clicked
if st.button("Predicted Salary"):
    experience_input=np.array([[years_experience]])
    prediction=model.predict(experience_input)
    
    st.success(f"The Predicted salary for{years_experience} years of experience is: ${prediction[0]:,.2f}")
    
st.write("The model was trained  ")