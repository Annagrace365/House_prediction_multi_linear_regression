import streamlit as st
import joblib

model=joblib.load("logistic_regression_studyhours_model.pkl")
st.title("Student Pass/Fail based on Study Hours and Attendance")
area=st.number_input("Enter Area:", min_value=600.0 ,max_value=3000.0, value=600.0)
bedrooms=st.number_input("Enter no of bedrooms:", min_value=1 ,max_value=4, value=3)
floors=st.number_input("Enter no of floors:", min_value=0 ,max_value=10, value=1)

if st.button("Predict"):
  prediction=model.predict([[area,bedrooms,floors]])
  st.write("Predicted price(in lakhs): ",prediction)
